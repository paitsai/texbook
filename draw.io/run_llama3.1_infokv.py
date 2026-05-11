import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目根目录
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from infokv import enable_infokv, set_rpc_config
import argparse
import json
from tqdm import tqdm
import numpy as np
import random
import torch.distributed as dist

from datasets import Dataset

def load_jsonl_with_datasets(file_path, split=None):
    """使用HuggingFace datasets库加载"""
    dataset = Dataset.from_json(file_path)
    
    if split == "test":
        # 如果文件已经是测试集，直接返回
        return dataset
    else:
        # 如果需要分割，可以使用train_test_split
        # 假设你想分割一部分作为测试集
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        return dataset[split]  # 返回指定的split

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k", "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "llama3" in model_name.lower() or "llama-3" in model_name.lower():
        # Llama 3 格式
        messages = [
            {"role": "user", "content": prompt}
        ]
        return f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    
    elif "llama-2" in model_name:
        return f"[INST]{prompt}[/INST]"
    elif "qwen" in model_name.lower():
        messages=[{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return text
   
    else:
        # 默认情况：尝试使用 apply_chat_template，如果失败则返回原始prompt
        try:
            messages = [{"role": "user", "content": prompt}]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        except:
            return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(
    data,
    args,
    dataset,
):

    tokenizer = args.tokenizer
    for item in tqdm(data):
        prompt = args.prompt_format.format(**item)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        max_length = args.max_len
        
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if args.dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, args.model_name)
        
        # prompt = build_chat(tokenizer, prompt, args.model_name)
        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        context_length = input.input_ids.shape[-1]
        max_gen = args.max_gen
        model = args.model
        
        if (
            args.dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
        else:
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0, # Qwen usually works with temp logic similar to other chats
            )[0]
        pred = output
        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, args.model_name)
        print(f"最终输出结果是：{pred}")
        with open(args.out_path, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "pred": pred,
                    "answers": item["answers"],
                    "all_classes": item["all_classes"],
                    "length": item["length"],
                    "_id": item["_id"],
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")
        

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "llama" in model_name:
        print(f"加载模型: {path}")
        mapping_device = get_layerwise_device_map(
            num_hidden_layers=32,  # Llama 3.1 8B 的 hidden layers 数量
            num_gpus=torch.cuda.device_count()-1,
            first_gpu=0,
            embed_on_first=True,
            lm_head_on_last=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=mapping_device,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    model = model.eval()
    return model, tokenizer

def load_hamlet_prompt(file_path="hamlet.txt", max_words=4000):
    """读取 Hamlet 前 max_words 个单词作为 prompt"""
    with open(file_path, "r") as f:
        text = f.read().strip()
    words = text.split()[:max_words]
    prompt = ' '.join(words)
    print(f"Prompt 单词数: {len(words)}, 字符数: {len(prompt)}")
    return prompt


def get_layerwise_device_map(
    num_hidden_layers: int,
    num_gpus: int,
    first_gpu: int = 0,
    embed_on_first: bool = True,
    lm_head_on_last: bool = True
) -> dict:
    """
    生成按层均匀分配的 device_map。
    
    Args:
        num_hidden_layers: 模型的 hidden layers 数量（例如 Llama-7B 为 32）
        num_gpus: 可用的 GPU 数量
        first_gpu: 起始 GPU 编号，默认为 0
        embed_on_first: 是否将 embedding 放在第一张卡（默认 True）
        lm_head_on_last: 是否将 lm_head 放在最后一张卡（默认 True）
    
    Returns:
        device_map 字典，例如：
        {
            'model.embed_tokens': 0,
            'model.layers.0': 0,
            'model.layers.1': 0,
            ...
            'model.layers.15': 1,
            ...
            'model.norm': last_gpu,
            'lm_head': last_gpu,
        }
    """
    device_map = {}
    
    # 计算每张卡分到的层数
    layers_per_gpu = num_hidden_layers // num_gpus
    remainder = num_hidden_layers % num_gpus
    
    # 分配 layers
    layer_idx = 0
    for gpu_id in range(num_gpus):
        # 当前 GPU 负责的层数
        n_layers = layers_per_gpu + (1 if gpu_id < remainder else 0)
        for _ in range(n_layers):
            device_map[f'model.layers.{layer_idx}'] = first_gpu + gpu_id
            layer_idx += 1
    
    # Embedding 层：通常放第一张卡
    if embed_on_first:
        device_map['model.embed_tokens'] = first_gpu
        device_map['model.embed_tokens.weight'] = first_gpu   # 某些模型结构可能需要
    else:
        device_map['model.embed_tokens'] = first_gpu + num_gpus - 1
    
    # Norm 和 lm_head：放最后一张卡
    last_gpu = first_gpu + num_gpus - 1
    device_map['model.norm'] = last_gpu
    device_map['lm_head'] = last_gpu + 1
    
    return device_map


def main():
    # ========== 配置参数 ==========
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", type=str, default="/home/users/xzr/llama-3.1-8b-instruct") 
    parser.add_argument("--max_len", type=int ,default=32000) 
    parser.add_argument("--n_proc", "-n", type=int, default=16)
    parser.add_argument("--P", type=int, default=1024)
    parser.add_argument("--R", type=int, default=64)
    parser.add_argument("--c", type=int, default=256, help="chunked compress ratio")
    parser.add_argument("--selectors", type=str, default="recent")
    parser.add_argument("--aggregation", type=str, default="all")
    parser.add_argument("--kernel_size", type=int, default=9)
    parser.add_argument("--pooling", type=str, default="avgpool")
    parser.add_argument("--pattern", type=str, default="uniform")
    parser.add_argument("--S", type=int, default=0)
    parser.add_argument("--layer_group_size", type=int, default=4)
    parser.add_argument("--score_pattern", type=str, default="entropy")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--data_dir", type=str, default="/home/users/xzr/benchmark/longbench-v1/data")
    parser.add_argument("--budget", type=int, default=-1, help="Budget for InfoKV, used for naming output folder")
    
    parser.add_argument(
        "--datasets", "-d", 
        type=str, 
        nargs='+', 
        default=[
            "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "qasper", 
            "2wikimqa", "narrativeqa", "samsum", "musique", "triviaqa", 
            "trec", "passage_retrieval_en", "passage_retrieval_zh", "passage_count"
        ],
        help="Space-separated list of datasets to process"
    )
    args = parser.parse_args()
    attn_implementation = "flash_attention_2"
    seed_everything(42)
    # 2. 启用 InfoKV（会 monkey-patch 模型）
    enable_infokv()
    
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.model_name, device="cuda")
    args.tokenizer = tokenizer
    args.model = model
    
    # 3. 加载模型和 tokenizer
    print(f"加载模型: {args.model_name}")
    # 4. 配置 InfoKV 参数
    set_rpc_config(
        model=model,
        P=args.P,
        R=args.R,
        c=args.c,
        selectors=args.selectors,
        aggregation=args.aggregation,
        kernel_size=args.kernel_size,
        pooling=args.pooling,
    )
    # 额外需要设置的 InfoKV 特有参数（根据原脚本）
    model.model.config.P = args.P
    model.model.config.R = args.R
    model.model.config.c = args.c
    model.model.config.pattern = args.pattern
    model.model.config.kernel_size = args.kernel_size
    model.model.config.pooling = args.pooling
    model.model.config.S = args.S
    model.model.config.layer_group_size = args.layer_group_size
    model.model.config.score_pattern = args.score_pattern
    model.model.config.tau = args.tau
    print("InfoKV 配置完成")

    # 5. Tokenize 并生成
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    if not os.path.exists("longbench"):
        os.makedirs("longbench")
        
    # load data
    for dataset in args.datasets:
        data_dir = args.data_dir
        model_name = args.model_name
        data = load_jsonl_with_datasets(f"{data_dir}/{dataset}.jsonl", split="test")
        data = list(data)
        
        # [修改] 使用 budget 命名输出文件夹
        if not os.path.exists(f"longbench/{model_name}-{args.budget}"):
            os.makedirs(f"longbench/{model_name}-{args.budget}")
        out_path = f"longbench/{model_name}-{args.budget}/{dataset}.jsonl"
        
        has_data = {}
        if os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                has_data = {json.loads(line)["_id"]: True for line in f}
                print(f"已存在 {len(has_data)} 条记录于 {out_path}，将跳过这些记录的预测。")
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        args.prompt_format = prompt_format
        args.max_gen = max_gen
        args.dataset = dataset
        args.out_path = out_path
        
        data_all = [data_sample for data_sample in data]
        new_data = []
        fout = open(out_path, "a", encoding="utf-8")
        for data_sample in data_all:
            if data_sample["_id"] not in has_data:
                new_data.append(data_sample)
        
        get_pred(
            new_data,
            args,
            dataset,
        )
        fout.close()
               


    
if __name__ == "__main__":
    main()