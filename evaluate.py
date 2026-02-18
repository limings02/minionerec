
import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig,  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper
from data import  EvalD3Dataset, EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor
from accelerate import Accelerator
import random
import bitsandbytes as bnb



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
P = 998244353
MOD = int(1e9 + 9)
import numpy as np

def get_hash(x):
    # 用tuple 作为dict key，更快，没有字符串开销
    return tuple(int(_) for _ in x)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    
def main(
    base_model: str = "",
    train_file: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float=0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
):
    random.seed(seed)
    set_seed(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    category_dict = {"Industrial_and_Scientific": "industrial and scientific items", "Office_Products": "office products", "Toys_and_Games": "toys and games", "Sports": "sports and outdoors", "Books": "books"}
    category = category_dict[category]
    print(category)

    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    with open(info_file, 'r') as f:
        info = f.readlines()
        # Parse new format: semantic_id \t item_title \t item_id
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
        
        # Format for tokenization
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
        info_titles = [f'''### Response:\n{_}''' for _ in item_titles]


    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Create prefixID for semantic IDs (existing functionality)
    if base_model.lower().find("llama") > -1:
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4
    else:
        prefix_index = 3
    
    # Build hash_dict for semantic IDs (existing functionality)
    hash_dict = dict()
    # print(f"eos token: {tokenizer.eos_token_id}")
    for index, ID in enumerate(prefixID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Build hash_dict_title for item titles (new functionality)
    hash_dict_title = dict()
    for index, ID in enumerate(prefixTitleID):
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict_title:
                hash_dict_title[hash_number] = set()
            hash_dict_title[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Convert sets to lists for both dictionaries
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    for key in hash_dict_title.keys():
        hash_dict_title[key] = list(hash_dict_title[key])

    # Define prefix constraint functions
    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []
        
    def prefix_allowed_tokens_fn_title(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict_title:
            return hash_dict_title[hash_number]
        return []

    # Default to semantic constraints (backward compatibility)
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
    # prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_title
    
    tokenizer.pad_token = tokenizer.eos_token #许多模型没有原生的pad_token，或者pad_token和eos_token是一样的，所以这里把pad_token设置成eos_token，这样在生成的时候就会用eos_token来填充，保证生成的合理性
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left" #pad放到左边，保证不同样本的末尾是对齐的。这里和我们的logits processor设计有关，我们的logits processor是根据输入的末尾来判断下一步生成的token应该是什么，所以需要保证输入的末尾是对齐的，这样才能正确地应用约束。
    
    # val_dataset = EvalD3Dataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
    val_dataset = EvalSidDataset(train_file=test_data_path, tokenizer=tokenizer, max_len=2560, category=category, test=True, K=K, seed=seed)
        
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    # encodings = [val_dataset[i] for i in indexes]
    test_data = val_dataset.get_all() #这个用的是我们内置的EvalDataset的get_all方法（其实是BaseDataset的）

    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate(
            encodings,
            num_beams=10,
            max_new_tokens=64,
            length_penalty=1.0,
            **kwargs,
    ):
        maxLen = max([len(_["input_ids"]) for _ in encodings]) #先统计最大长度

        padding_encodings = {"input_ids": []}
        attention_mask = []

        for  _ in encodings:
            L = len(_["input_ids"])
            padding_encodings["input_ids"].append([tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"])
            attention_mask.append([0] * (maxLen - L) + [1] * L)  #手动实现左pad
        
        # print(f"num_beams: {num_beams}")
        generation_config = GenerationConfig(
            num_beams=num_beams,
            length_penalty=length_penalty,
            num_return_sequences=num_beams, #num_beams和num_return_sequences都设置成num_beams，表示每个输入生成num_beams个输出，这样就可以得到每个输入对应的num_beams个候选答案，方便后续评估
            pad_token_id = model.config.pad_token_id,
            eos_token_id = model.config.eos_token_id,
            max_new_tokens = max_new_tokens,
            top_k=None,
            top_p=None,
            **kwargs
        )
        
        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model,
                eos_token_id=model.config.eos_token_id
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),
                attention_mask=torch.tensor(attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )
        # generate()返回的sequences是prompt+completion拼在一起的，我们只想要模型生成的部分SID，所以把前maxlen切掉（已经pad过了）
        # 这也是左pad的方便之处，保证了输入的末尾是对齐的，这样切掉前maxLen就能得到模型生成的部分
        batched_completions = generation_output.sequences[:, maxLen:]
        
        
        if base_model.lower().find("llama") > -1:
            # batch_decode默认会把token id转换成字符串的时候去掉一些特殊字符，这些特殊字符在llama的tokenizer里是有特殊含义的，所以我们需要保留这些特殊字符，不能让batch_decode去掉它们，所以设置clean_up_tokenization_spaces=False，这样就能保留这些特殊字符，保证生成的合理性
            # skip_special_tokens=True表示在解码的时候去掉特殊token，clean_up_tokenization_spaces=False表示在解码的时候不去掉多余的空格，这样就能保留生成文本中的特殊格式，保证生成的合理性
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            output = tokenizer.batch_decode(batched_completions, skip_special_tokens=True)
            
        output = [_.split("Response:\n")[-1].strip() for _ in output] #其实前面已经把那部分截掉了，这里再做一次保险，确保只保留生成的部分，去掉可能存在的前缀信息
        # 下面是把扁平列表恢复成每个样本对应num_beams个候选答案的列表，方便后续评估
        # 最后想要存到json的是每个样本对应一个候选答案列表的格式，所以要把扁平列表恢复成每个样本对应num_beams个候选答案的格式
        real_outputs = [output[i * num_beams: (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs #这就是最后的输出格式，每个元素是一个样本对应的num_beams个候选答案的列表
    
    model = model.to(device)

    from tqdm import tqdm
    outputs = []
    new_encodings = []
    BLOCK = (len(encodings) + batch_size - 1) // batch_size
    for i in range(BLOCK):
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size]) #这里在拼batch

    
    for idx, encodings in enumerate(tqdm(new_encodings)):
        # Use standard evaluation
        output = evaluate(encodings, max_new_tokens=max_new_tokens, num_beams=num_beams, length_penalty=length_penalty)
        # 这里先分batch生成输出，然后把每个batch的输出拼接在一起，得到最终的outputs列表，里面每个元素是一个样本对应的num_beams个候选答案的列表
        outputs = outputs + output
        
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i] # 给test加一个predict字段，存放模型生成的num_beams个候选答案的列表
    

    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')  # 这个dedup字段是之前评估过程中用来判断是否去重的，现在不需要了，所以删除掉，避免存到json里造成混乱
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)





