import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torchdynamo_compiling,
    logging,
    replace_return_docstrings,
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv, LLAMA_INPUTS_DOCSTRING, LlamaFlashAttention2
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from infokv.rpc_utils import init_rpc


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


import time

def _sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def _get_layer_groups(num_hidden_layers: int, layer_group_size: int) -> List[List[int]]:
    layer_group_size = max(int(layer_group_size or 1), 1)
    return [
        list(range(group_start, min(group_start + layer_group_size, num_hidden_layers)))
        for group_start in range(0, num_hidden_layers, layer_group_size)
    ]


def _compute_layer_logits(
    model,
    layer_hidden_states: torch.Tensor,
    lm_head_slices: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    if model.config.pretraining_tp > 1:
        layer_logits = [F.linear(layer_hidden_states, lm_head_slices[i]) for i in range(model.config.pretraining_tp)]
        layer_logits = torch.cat(layer_logits, dim=-1)
    else:
        layer_logits = model.lm_head(layer_hidden_states)
    return layer_logits.float()


# def _compute_token_score(layer_logits: torch.Tensor, score_pattern: str) -> torch.Tensor:
#     if score_pattern == "prob_margin":
#         probs = F.softmax(layer_logits, dim=-1)
#         top2 = torch.topk(probs, k=2, dim=-1)
#         return 1 / (top2.values[..., 0] - top2.values[..., 1] + 1e-8)
#     if score_pattern == "prob_margin_multi":
#         probs = F.softmax(layer_logits, dim=-1)
#         top2 = torch.topk(probs, k=2, dim=-1)
#         return 1 / (top2.values[..., 0] * (top2.values[..., 0] - top2.values[..., 1] + 1e-8))
#     if score_pattern == "entropy":
#         return -(F.log_softmax(layer_logits, dim=-1) * F.softmax(layer_logits, dim=-1)).sum(dim=-1)
#     raise ValueError(f"Unsupported score_pattern: {score_pattern}")
import torch
import torch.nn.functional as F
from typing import Optional

def _compute_token_score(
    layer_logits: torch.Tensor, 
    score_pattern: str,
    chunk_size: Optional[int] = None
) -> torch.Tensor:
    """
    Compute token scores with optional chunked processing for large logits.
    
    Args:
        layer_logits: Tensor of shape (..., vocab_size)
        score_pattern: One of "prob_margin", "prob_margin_multi", "entropy"
        chunk_size: If provided, process vocab dimension in chunks.
                   Default None means no chunking.
    """
    # Auto-determine chunk size if not specified
    if chunk_size is None:
        vocab_size = layer_logits.shape[-1]
        # Only chunk if vocab is large (> 50000)
        if vocab_size > 50000:
            # Estimate chunk size based on available memory
            chunk_size = min(10000, vocab_size // 4)
        else:
            chunk_size = vocab_size  # No chunking needed
    
    if chunk_size >= layer_logits.shape[-1]:
        # No chunking needed
        return _compute_full(layer_logits, score_pattern)
    
    # Chunked computation
    return _compute_chunked(layer_logits, score_pattern, chunk_size)


def _compute_full(layer_logits: torch.Tensor, score_pattern: str) -> torch.Tensor:
    """Original computation without chunking for small/regular vocab sizes."""
    if score_pattern == "prob_margin":
        probs = F.softmax(layer_logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1)
        return 1 / (top2.values[..., 0] - top2.values[..., 1] + 1e-8)
    
    elif score_pattern == "prob_margin_multi":
        probs = F.softmax(layer_logits, dim=-1)
        top2 = torch.topk(probs, k=2, dim=-1)
        return 1 / (top2.values[..., 0] * (top2.values[..., 0] - top2.values[..., 1] + 1e-8))
    
    elif score_pattern == "entropy":
        log_probs = F.log_softmax(layer_logits, dim=-1)
        probs = F.softmax(layer_logits, dim=-1)
        return -(log_probs * probs).sum(dim=-1)
    
    raise ValueError(f"Unsupported score_pattern: {score_pattern}")


def _compute_chunked(
    layer_logits: torch.Tensor, 
    score_pattern: str, 
    chunk_size: int
) -> torch.Tensor:
    """
    Chunked computation for large vocab sizes.
    """
    vocab_size = layer_logits.shape[-1]
    
    if score_pattern == "entropy":
        return _chunked_entropy(layer_logits, chunk_size)
    elif score_pattern in ["prob_margin", "prob_margin_multi"]:
        return _chunked_prob_margin(layer_logits, score_pattern, chunk_size)
    else:
        raise ValueError(f"Unsupported score_pattern: {score_pattern}")


def _chunked_entropy(layer_logits: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """
    Compute entropy using chunked log-sum-exp trick for numerical stability.
    
    entropy = -sum(p_i * log(p_i))
           = log(sum(exp(x_i))) - sum(x_i * exp(x_i)) / sum(exp(x_i))
    """
    vocab_size = layer_logits.shape[-1]
    original_shape = layer_logits.shape
    
    # Keep all batch dimensions
    batch_dims = original_shape[:-1]
    
    # Step 1: Compute max for numerical stability
    max_logits = layer_logits.max(dim=-1, keepdim=True).values  # (..., 1)
    
    # Step 2: Chunked computation of Z and weighted_sum
    Z = torch.zeros(batch_dims + (1,), device=layer_logits.device, dtype=layer_logits.dtype)
    weighted_sum = torch.zeros(batch_dims + (1,), device=layer_logits.device, dtype=layer_logits.dtype)
    
    for i in range(0, vocab_size, chunk_size):
        end_idx = min(i + chunk_size, vocab_size)
        chunk = layer_logits[..., i:end_idx]
        
        exp_chunk = torch.exp(chunk - max_logits)
        Z += exp_chunk.sum(dim=-1, keepdim=True)
        
        # For entropy: sum(x_i * exp(x_i))
        weighted_sum += (chunk * exp_chunk).sum(dim=-1, keepdim=True)
    
    # Step 3: Compute log(Z) and entropy
    log_Z = torch.log(Z) + max_logits  # (..., 1)
    
    # entropy = log_Z - weighted_sum / Z
    entropy = log_Z.squeeze(-1) - (weighted_sum.squeeze(-1) / Z.squeeze(-1))
    
    # entropy should now have shape batch_dims
    return entropy


def _chunked_prob_margin(
    layer_logits: torch.Tensor, 
    score_pattern: str, 
    chunk_size: int
) -> torch.Tensor:
    """
    Compute prob_margin scores with chunked processing.
    """
    vocab_size = layer_logits.shape[-1]
    original_shape = layer_logits.shape
    batch_dims = original_shape[:-1]
    
    # Flatten for easier processing if needed
    if len(batch_dims) > 1:
        flat_logits = layer_logits.reshape(-1, vocab_size)
        batch_size = flat_logits.shape[0]
        
        # Process and then reshape back
        result = _chunked_prob_margin_2d(flat_logits, score_pattern, chunk_size)
        return result.reshape(batch_dims)
    else:
        return _chunked_prob_margin_general(layer_logits, score_pattern, chunk_size)


def _chunked_prob_margin_general(
    layer_logits: torch.Tensor, 
    score_pattern: str, 
    chunk_size: int
) -> torch.Tensor:
    """General version handling arbitrary batch dimensions."""
    vocab_size = layer_logits.shape[-1]
    batch_dims = layer_logits.shape[:-1]
    
    # Initialize top-2 values
    top2_values = torch.full(
        batch_dims + (2,), 
        float('-inf'), 
        device=layer_logits.device, 
        dtype=layer_logits.dtype
    )
    
    # Find global top-2 logits
    for i in range(0, vocab_size, chunk_size):
        end_idx = min(i + chunk_size, vocab_size)
        chunk = layer_logits[..., i:end_idx]
        
        # Get top-2 from this chunk
        chunk_top2 = torch.topk(chunk, k=min(2, chunk.shape[-1]), dim=-1)
        
        # Merge with global top-2
        combined = torch.cat([top2_values, chunk_top2.values], dim=-1)
        top2_values = torch.topk(combined, k=2, dim=-1).values
    
    # Compute Z = sum(exp(logits - max))
    max_logit = top2_values[..., 0:1]  # (..., 1)
    Z = torch.zeros(batch_dims + (1,), device=layer_logits.device, dtype=layer_logits.dtype)
    
    for i in range(0, vocab_size, chunk_size):
        end_idx = min(i + chunk_size, vocab_size)
        chunk = layer_logits[..., i:end_idx]
        Z += torch.exp(chunk - max_logit).sum(dim=-1, keepdim=True)
    
    # Compute probabilities for top-2
    top2_probs = torch.exp(top2_values - max_logit) / Z
    
    # Calculate score
    diff = top2_probs[..., 0] - top2_probs[..., 1] + 1e-8
    
    if score_pattern == "prob_margin":
        score = 1 / diff
    else:  # prob_margin_multi
        score = 1 / (top2_probs[..., 0] * diff)
    
    return score


def _chunked_prob_margin_2d(
    flat_logits: torch.Tensor,  # (batch_size, vocab_size)
    score_pattern: str, 
    chunk_size: int
) -> torch.Tensor:
    """Optimized version for 2D input."""
    batch_size, vocab_size = flat_logits.shape
    
    # Initialize top-2 values
    top2_values = torch.full(
        (batch_size, 2), 
        float('-inf'), 
        device=flat_logits.device, 
        dtype=flat_logits.dtype
    )
    
    # Find global top-2 logits
    for i in range(0, vocab_size, chunk_size):
        end_idx = min(i + chunk_size, vocab_size)
        chunk = flat_logits[:, i:end_idx]
        
        chunk_top2 = torch.topk(chunk, k=min(2, chunk.shape[-1]), dim=-1)
        
        combined = torch.cat([top2_values, chunk_top2.values], dim=-1)
        top2_values = torch.topk(combined, k=2, dim=-1).values
    
    # Compute Z
    max_logit = top2_values[:, 0:1]
    Z = torch.zeros(batch_size, 1, device=flat_logits.device, dtype=flat_logits.dtype)
    
    for i in range(0, vocab_size, chunk_size):
        end_idx = min(i + chunk_size, vocab_size)
        chunk = flat_logits[:, i:end_idx]
        Z += torch.exp(chunk - max_logit).sum(dim=-1, keepdim=True)
    
    # Compute probabilities
    top2_probs = torch.exp(top2_values - max_logit) / Z
    
    # Calculate score
    diff = top2_probs[:, 0] - top2_probs[:, 1] + 1e-8
    
    if score_pattern == "prob_margin":
        return 1 / diff
    else:
        return 1 / (top2_probs[:, 0] * diff)

def _expand_group_values(group_values: torch.Tensor, layer_groups: List[List[int]], divide_by_group_size: bool = False) -> torch.Tensor:
    expanded_values = []
    for group_value, group in zip(group_values, layer_groups):
        per_layer_value = group_value / len(group) if divide_by_group_size else group_value
        expanded_values.extend([per_layer_value] * len(group))

    return torch.stack(expanded_values)


def _allocate_grouped_retain_counts(
    group_ratios: torch.Tensor,
    layer_groups: List[List[int]],
    total_retain_count: int,
    max_per_layer: int,
) -> List[int]:
    group_ratios = group_ratios.float()
    ratio_sum = group_ratios.sum()
    if ratio_sum <= 0:
        group_ratios = torch.full_like(group_ratios, 1.0 / group_ratios.numel())
    else:
        group_ratios = group_ratios / ratio_sum

    group_lengths = torch.tensor([len(group) for group in layer_groups], device=group_ratios.device, dtype=group_ratios.dtype)
    per_group_budget = group_ratios * float(total_retain_count)
    group_counts = torch.floor(per_group_budget / group_lengths).long().clamp(min=1, max=max_per_layer)

    expanded_counts = []
    for group_count, group in zip(group_counts.tolist(), layer_groups):
        expanded_counts.extend([group_count] * len(group))

    leftover = int(total_retain_count - sum(expanded_counts))
    group_idx = 0
    while leftover > 0:
        progressed = False
        for _ in range(len(layer_groups)):
            group_len = len(layer_groups[group_idx])
            if group_counts[group_idx] < max_per_layer and leftover >= group_len:
                group_counts[group_idx] += 1
                leftover -= group_len
                progressed = True
            group_idx = (group_idx + 1) % len(layer_groups)
            if leftover <= 0:
                break
        if not progressed:
            break

    expanded_counts = []
    for group_count, group in zip(group_counts.tolist(), layer_groups):
        expanded_counts.extend([group_count] * len(group))

    return expanded_counts


class LlamaRPCAttention(LlamaFlashAttention2):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        init_rpc(self)
        self.verbose = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # Record attention-based importance into the RPCCluster for later use
        # This saves the cost of recomputing Q/K in the outer scoring logic.
        if q_len > 1:
            self.kv_cluster.prompt_len = past_key_value.get_seq_length(self.layer_idx)
            self.kv_cluster.num_comp = 0
            
        target_len = past_key_value.get_seq_length(self.layer_idx) - self.kv_cluster.prompt_len - (self.kv_cluster.num_comp * self.kv_cluster.T) - self.kv_cluster.R

        if target_len > self.kv_cluster.P - self.kv_cluster.R:
            self.kv_cluster.cache_recent(query_states)

        if target_len == self.kv_cluster.P:
            self.kv_cluster.compute_attn_weights(key_states)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask[:,:key_states.shape[1]] if attention_mask is not None else attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def llama_casual_forward_InfoKV(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    t0 = _sync_time()
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=return_dict,
        cache_position=cache_position,
    )
    
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)

    t1 = _sync_time()
    model_time = t1 - t0

    topk_time = 0.0
    kv_rewrite_time = 0.0
    entropy_update_time = 0.0
    decode_entropy_time = 0.0
    if input_ids.shape[1] > 1: # 这里是prefilling阶段
        # import pdb; pdb.set_trace()
        self.time_accum = {
            "model": 0.0,
            "topk": 0.0,
            "kv_rewrite": 0.0,
            "entropy_update": 0.0,
            "forward_total": 0.0,
            "decode_entropy": 0.0,
            "loss_compute": 0.0,
        }
        self.forward_count = 1
        self.time_accum["model"] += model_time

        self.current_len = input_ids.shape[1]
        self.prompt_len = input_ids.shape[1]
        print(f"Initial input length: {self.current_len}, prompt length: {self.prompt_len}")
        self.num_comp = 0

        B, T, _ = outputs[0].shape

        self.decoding_entropy_layers = []
        
        print(f"input_ids: {input_ids}")
        print(f"shape of input_ids: {input_ids.shape}, eos_token_id: {self.config.eos_token_id}")
        
        # mask_pad = (input_ids == self.config.eos_token_id)  # (B, T)
        # 确保 eos_token_id 是一个列表或张量
        eos_ids = self.config.eos_token_id
        if not isinstance(eos_ids, (list, tuple, torch.Tensor)):
            eos_ids = [eos_ids]
            
        eos_tensor = torch.tensor(eos_ids, device=input_ids.device)
        mask_pad = torch.isin(input_ids, eos_tensor)  # shape (B, T) bool
        
        layer_group_size = getattr(self.config, "layer_group_size", 1)
        self.layer_groups = _get_layer_groups(self.config.num_hidden_layers, layer_group_size)

        group_score_layers = []
        for layer_group in self.layer_groups:
            layer_idx = layer_group[-1]
            layer_hidden_states = outputs.hidden_states[layer_idx + 1][:, self.config.S:]  # exclude the first S tokens from entropy computation
            layer_logits = _compute_layer_logits(self, layer_hidden_states, lm_head_slices if self.config.pretraining_tp > 1 else None)
            score = _compute_token_score(layer_logits, self.config.score_pattern)

            if mask_pad.any():
                pad_mask = mask_pad[:, self.config.S:]
                score = score.masked_fill(pad_mask.to(score.device), 0.0)

            group_score_layers.append(score.sum())
            grouped_decode_score = score[:, -1].unsqueeze(-1)
            for _ in layer_group:
                self.decoding_entropy_layers.append(grouped_decode_score.clone())

        self.group_score_layers = torch.stack(group_score_layers)
        if self.config.pattern == "uniform":
            self.retain_count_layers = [(self.num_comp + 1) * self.config.P // self.config.c] * self.config.num_hidden_layers
        elif self.config.pattern == "adaptive":
            total_retain_count = (self.config.P // self.config.c) * self.config.num_hidden_layers
            self.group_score_layers = torch.log((self.group_score_layers + 1)/self.config.tau)
            score_sum = self.group_score_layers.sum()
            if score_sum > 0:
                self.retain_ratio_groups = self.group_score_layers / score_sum
            else:
                self.retain_ratio_groups = torch.full_like(
                    self.group_score_layers,
                    1.0 / self.group_score_layers.numel(),
                )
            self.retain_ratio_layers = _expand_group_values(
                self.retain_ratio_groups,
                self.layer_groups,
                divide_by_group_size=True,
            )
            self.retain_count_layers = _allocate_grouped_retain_counts(
                self.retain_ratio_groups,
                self.layer_groups,
                total_retain_count,
                self.config.P,
            )

            self.per_retain_count_layers = [retain_count for retain_count in self.retain_count_layers]

        for layer_idx in range(self.config.num_hidden_layers):
            self.model.layers[layer_idx].self_attn.kv_cluster.T = self.retain_count_layers[layer_idx]
        # print(f"Initial retain_count_layers: {self.retain_count_layers}")
        # import pdb; pdb.set_trace()
        t2 = _sync_time()
        decode_entropy_time += t2 - t1
        self.time_accum["decode_entropy"] += decode_entropy_time
        
    else:
        self.current_len += 1
        self.forward_count += 1
        self.time_accum["model"] += model_time

        target_len = self.current_len - self.prompt_len - (self.num_comp * self.config.P // self.config.c) - self.config.R

        if target_len == self.config.P:
            keep_indices_layers = []
            for layer_idx in range(self.config.num_hidden_layers):
                t3 = _sync_time()
                retain_count = self.retain_count_layers[layer_idx]
                entropy = self.decoding_entropy_layers[layer_idx][:, :-self.config.R]  # (B, cached_len-R)
                entropy = nn.functional.softmax(entropy, dim=-1)  # (B, cached_len-R)
                attn_weights = self.model.layers[layer_idx].self_attn.kv_cluster.attn_weights_sum
                # if layer_idx == 0:
                #     print(f"entropy shape: {entropy.shape}, attn_weights shape: {attn_weights.shape}, past_key_values shape: {outputs.past_key_values[layer_idx][0].shape}, prompt_len: {self.prompt_len}, {self.model.layers[layer_idx].self_attn.kv_cluster.prompt_len}, {self.model.layers[layer_idx].self_attn.kv_cluster.num_comp}")

                score = entropy + attn_weights.to(entropy.device)  # (B, cached_len-R)
                # score = attn_weights  # (B, cached_len-R)
                if self.config.pooling == 'avgpool':
                    score = F.avg_pool1d(score, kernel_size = self.config.kernel_size, padding=self.config.kernel_size//2, stride=1)
                elif self.config.pooling == 'maxpool':
                    score = F.max_pool1d(score, kernel_size = self.config.kernel_size, padding=self.config.kernel_size//2, stride=1)

                # compress tokens with the lowest score
                keep_indices = torch.topk(score, k=retain_count, dim=1, largest=True).indices.sort(dim=-1).values  # (B, k)
                t4 = _sync_time()
                topk_time += t4 - t3

                layer_cache = outputs.past_key_values[layer_idx]
                # --- read ---
                if hasattr(layer_cache, "key"):
                    keys = layer_cache.key
                    values = layer_cache.value
                else:
                    keys, values = layer_cache
                    
                K = keys.shape[2]

                assert keep_indices.min() >= 0
                assert keep_indices.max() < K, \
                    f"max index {keep_indices.max()} >= kv_seq_len {K}"

                # --- gather ---
                gather_keep_indices = keep_indices.to(keys.device).unsqueeze(1).unsqueeze(-1).expand(-1, keys.size(1), -1, keys.size(-1))
                compressed_keys = keys[:, :, self.prompt_len:-self.config.R].gather(dim=2, index=gather_keep_indices)
                compressed_values = values[:, :, self.prompt_len:-self.config.R].gather(dim=2, index=gather_keep_indices)

                # --- write back ---
                outputs.past_key_values.key_cache[layer_idx] = torch.cat([keys[:, :, :self.prompt_len, :], compressed_keys, keys[:, :, -self.config.R:, :]], dim=2).contiguous()
                outputs.past_key_values.value_cache[layer_idx] = torch.cat([values[:, :, :self.prompt_len, :], compressed_values, values[:, :, -self.config.R:, :]], dim=2).contiguous()

                # import pdb; pdb.set_trace()
                
                self.decoding_entropy_layers[layer_idx] = torch.cat([self.decoding_entropy_layers[layer_idx].gather(dim=1, index=keep_indices), self.decoding_entropy_layers[layer_idx][:, -self.config.R:]], dim=1)
                t5 = _sync_time()
                kv_rewrite_time += t5 - t4
            
            t6 = _sync_time()
            
            self.num_comp += 1
            self.current_len = self.prompt_len + (self.num_comp * self.config.P // self.config.c) + self.config.R

            if self.config.pattern == "uniform":
                self.retain_count_layers = [(self.num_comp + 1) * self.config.P // self.config.c] * self.config.num_hidden_layers
            elif self.config.pattern == "adaptive":
                self.retain_count_layers = [(self.num_comp + 1) * retain_count for retain_count in self.per_retain_count_layers]

            t7 = _sync_time()
            entropy_update_time += t7 - t6
            self.time_accum["entropy_update"] += entropy_update_time
            self.time_accum["kv_rewrite"] += kv_rewrite_time
            self.time_accum["topk"] += topk_time

        t8 = _sync_time()
        
     
        eos_ids = self.config.eos_token_id
        if not isinstance(eos_ids, (list, tuple, torch.Tensor)):
            eos_ids = [eos_ids]
        eos_tensor = torch.tensor(eos_ids, device=input_ids.device)
        mask_pad_2 = torch.isin(input_ids, eos_tensor)  # shape (B, T) bool
        
        mask_eos = mask_pad_2.any(dim=1)  # (B,)
        for layer_group in self.layer_groups:
            layer_idx = layer_group[-1]
            layer_hidden_states = outputs.hidden_states[layer_idx + 1]
            layer_logits = _compute_layer_logits(self, layer_hidden_states, lm_head_slices if self.config.pretraining_tp > 1 else None)
            score = _compute_token_score(layer_logits, self.config.score_pattern)

            # Mask out the entropy for sequences that have already finished (i.e., have generated an EOS token)
            if mask_eos.any():
                eos_mask = mask_eos.unsqueeze(-1)  # (B, 1)
                score = score.masked_fill(eos_mask.to(score.device), 0.0)

            for grouped_layer_idx in layer_group:
                self.decoding_entropy_layers[grouped_layer_idx] = torch.cat(
                    [self.decoding_entropy_layers[grouped_layer_idx], score],
                    dim=1,
                )  # (B, cached_len+1)
        
        t9 = _sync_time()
        decode_entropy_time += t9 - t8
        self.time_accum["decode_entropy"] += decode_entropy_time

    t10 = _sync_time()
    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
        logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
        logits = torch.cat(logits, dim=-1)
    else:
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

    loss = None
    if labels is not None:
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
    
    t11 = _sync_time()
    loss_compute_time = t11 - t10
    self.time_accum["loss_compute"] += loss_compute_time
    forward_time = t11 - t0
    self.time_accum["forward_total"] += forward_time

    if self.current_len == self.prompt_len + (self.num_comp * self.config.P // self.config.c) + self.config.R:
        print(
            f"\n[Forward {self.forward_count}] "
            f"step={self.current_len}, "
            f"compression num={self.num_comp}\n"
            f"  This forward (ms):\n"
            f"    model         : {model_time*1000:.2f}\n"
            f"    topk          : {topk_time*1000:.2f}\n"
            f"    kv_rewrite    : {kv_rewrite_time*1000:.2f}\n"
            f"    entropy_update: {entropy_update_time*1000:.2f}\n"
            f"    decode_entropy: {decode_entropy_time*1000:.2f}\n"
            f"    loss_compute  : {loss_compute_time*1000:.2f}\n"
            f"    total         : {forward_time*1000:.2f}\n"
            f"  Accumulated (ms):\n"
            f"    model         : {self.time_accum['model']*1000:.2f}\n"
            f"    topk          : {self.time_accum['topk']*1000:.2f}\n"
            f"    kv_rewrite    : {self.time_accum['kv_rewrite']*1000:.2f}\n"
            f"    entropy_update: {self.time_accum['entropy_update']*1000:.2f}\n"
            f"    decode_entropy: {self.time_accum['decode_entropy']*1000:.2f}\n"
            f"    loss_compute  : {self.time_accum['loss_compute']*1000:.2f}\n"
            f"    total         : {self.time_accum['forward_total']*1000:.2f}\n"
            f"  retain_count_layers: {self.retain_count_layers}\n"
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
