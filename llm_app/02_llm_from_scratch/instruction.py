import os
import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
from device import common_device
from model import GPTModel
import json
import urllib.request

tokenizer = tiktoken.get_encoding("gpt2")

# ================================ 准备数据集 ================================
# 下载并加载数据
def load_data(file_name = "instruction/instruction-data.json"):

    def download_and_load_file(file_path, url):
        import json
        import urllib.request

        if not os.path.exists(file_path):
            with urllib.request.urlopen(url) as response:
                text_data = response.read().decode("utf-8")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        
        return data

    # 组织数据存放目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
            "/main/ch07/01_main-chapter-code/instruction-data.json")
    
    # 下载并加载数据
    data = download_and_load_file(file_path, url)

    # print("Number of entries: ", len(data))
    # print("Example entry: \n", data[50])
    # print("Another example entry: \n", data[999])

    return data
    

# practice 7.1: 改变提示词风格，重新微调，对比回复质量

# 定义提示词模板函数，将样本转为Alpaca风格的输入格式
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describe a task. "
        f"Write a response that appropriatelly completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = (
        f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    )
    return instruction_text + input_text

# 测试提示词风格输入-输出
def test_format_input():
    data = load_data()

    # 带输入
    model_input = format_input(data[50])
    desired_reponse = f"\n\n### Response:\n{data[50]['output']}"
    print(model_input + desired_reponse)

    # 不带输入
    model_input = format_input(data[999])
    desired_reponse = f"\n\n### Response:\n{data[999]['output']}"
    print(model_input + desired_reponse)

# 划分数据集
def prepare_data():
    data = load_data()
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    return train_data, val_data, test_data
    print("Training set length: ", len(train_data))
    print("Validation set length: ", len(val_data))
    print("Test set length: ", len(test_data))

# ================================ 组织训练批次 ================================
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            instruction_text = format_input(entry)
            desired_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_text + desired_text
            self.encoded_texts.append(tokenizer.encode(full_text))
    
    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

# 自定义聚合函数：填充批次，将同批数据长度对齐
# 提问 1：为什么这里要把长度定为len(item) + 1
def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_list = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # 此处算出的padded的长度为实际序列最长长度+1
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1])
        inputs_list.append(inputs)
    
    inputs_tensor = torch.stack(inputs_list).to(device)
    return inputs_tensor

# 更新自定义聚合函数
def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_list, targets_list = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        # 此处算出的padded的长度为实际序列最长长度+1
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))

        # 组织输入输出：输出为输入左移一位
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        inputs_list.append(inputs)
        targets_list.append(targets)
    
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)
    return inputs_tensor, targets_tensor

# 提问 2：为什么只对目标序列进行填充词替换，又为什么替换为-100
# 自定义函数定版：填充、生成输入-目标对、替换目标序列填充
def custom_collate_fn(batch, pad_token_id=50256,
    ignore_index=-100, allowed_max_length=None, device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]

        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        
        # 替换目标序列特殊词元
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        
        # 若长度超出限制，需要截断
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor

# 测试自定义聚合函数 
def test_custom_collate():
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]
    batch = (inputs_1, inputs_2, inputs_3)
    inputs_tensor = custom_collate_draft_1(batch)
    print("Test custom_collate_draft_1, inputs: \n", inputs_tensor)

    inputs, targets = custom_collate_draft_2(batch)
    print("Test custom_collate_draft_2, inputs: \n", inputs)
    print("Test custom_collate_draft_2, targets: \n", targets)

    inputs, targets = custom_collate_fn(batch)
    print("Test custom_collate_fn, inputs: \n", inputs)
    print("Test custom_collate_fn, targets: \n", targets)

# practice7.2
def practice7_2():
    pass
    # 有两种思路
    # ① 在 custom_collate_fn() 的targets中直接找 ### Response的index，将之前的词元都置为-100。这种做法不严谨，有风险
    # ② 修改 InstructionDataset() 输出，把每个text的指令部分index直接返回。在custom_collate_fn() 中直接根据索引对targets做指令掩码

# ================================ 创建数据加载器 ================================
def create_instrucition_loader():
    from functools import partial

    # 使用偏函数：预设函数参数
    device = common_device
    customed_collect_fn = partial(
        custom_collate_fn,
        device = device,
        allowed_max_length = 1024
    )
    
    # 准备数据集
    train_data, val_data, test_data = prepare_data()
    train_dataset = InstructionDataset(train_data, tokenizer)
    val_dataset = InstructionDataset(val_data, tokenizer)
    test_dataset = InstructionDataset(test_data, tokenizer)
    num_workers = 0
    batch_size = 8

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size = batch_size,
        collate_fn=customed_collect_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size = batch_size,
        collate_fn=customed_collect_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size = batch_size,
        collate_fn=customed_collect_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

# 测试数据加载器
def test_instruction_loader():
    train_loader, val_loader, test_loader = create_instrucition_loader()
    print("Train loader: ")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)
    print(inputs)
    print(targets)

# ================================ 加载预训练模型 ================================
def load_pretrained_model():
    from gpt_download import download_and_load_gpt2
    from model import GPTModel
    from pre_train import load_weights_into_gpt, model_configs

    BASE_CONFIG = {
        "vocab_size":50257,
        "context_length":1024,
        "drop_rate":0.0,
        "qkv_bias":True
    }
    
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    _, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir=os.path.join(script_dir, "models")
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    return model

# 测试预训练模型验证集指令任务表现
def val_simple_model():
    model = load_pretrained_model()

    # 取一条输入
    # torch.manual_seed(123)
    _, val_data, _ = prepare_data()
    input_text = format_input(val_data[0])
    print(input_text)

    # 用刚加载的预训练模型生成回复
    from pre_train import generate, text_to_token_ids, token_ids_to_text
    token_ids = generate(
        model,
        idx = text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=1024,
        eos_id = 50256
    )
    gen_texts = token_ids_to_text(token_ids, tokenizer)
    print(gen_texts)

    # 抽取回复内容
    response_text = gen_texts[len(input_text):].strip()
    print(response_text)

# ================================ 指令微调大模型 ================================
# 提问 3：此处为什么初始loss计算与书上不同
# 计算初始损失
def calc_origin_loss():
    from pre_train import calc_loss_loader

    model = load_pretrained_model()
    device = common_device
    model.to(device)
    train_loader, val_loader, _ = create_instrucition_loader()

    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, 5)
        val_loss = calc_loss_loader(val_loader, model, device, 5)
    print("Origin training loss: ", train_loss)
    print("Origin validation loss: ", val_loss)

# 指令微调
def train_instruction_ft():
    import time
    from pre_train import train_model_simple, plot_losses

    model = load_pretrained_model()
    device = common_device
    model.to(device)
    _, val_data, _ = prepare_data()
    train_loader, val_loader, _ = create_instrucition_loader()

    start_time=time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
    num_epochs = 2

    # 训练主循环
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, 
        device, num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # 保存模型
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "models/test/model_instruction.pth")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)

#practice7.3：在Alpaca原始数据集进行微调
# 回去在cuda上跑

# 提问 4：指令微调与分类微调的区别
# ================================ 保存模型回复 ================================
# 打印对比模型回复与标准回复
def compare_responses():
    from pre_train import model_configs, generate, text_to_token_ids, token_ids_to_text
    _, _, test_data = prepare_data()
    
    # 加载模型
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 读取并加载模型权重
    model = GPTModel(BASE_CONFIG)

    # 加载模型文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "models/test/model_instruction.pth")
    state_dict = torch.load(filepath, map_location=common_device, weights_only=True)
    result = model.load_state_dict(state_dict)
    assert not result.missing_keys and not result.unexpected_keys, result
    device = common_device
    model.to(device)
    model.eval()

    torch.manual_seed(123)

    # 生成回复
    for entry in test_data[:3]:
        input_text = format_input(entry)
            
        token_ids = generate(
            model,
            idx = text_to_token_ids(input_text, tokenizer),
            max_new_tokens=256,
            context_size=1024,
            eos_id = 50256)

        gen_texts = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            gen_texts[len(input_text):].replace("### Response:", "").strip()
        )
        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("------------------------------------------------")

# 生成所有回复，并保存为文件
def generate_response():
    from tqdm import tqdm

    from pre_train import model_configs, generate, text_to_token_ids, token_ids_to_text
    _, _, test_data = prepare_data()
    
    # 加载模型
    CHOOSE_MODEL = "gpt2-medium (355M)"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 读取并加载模型权重
    model = GPTModel(BASE_CONFIG)

    # 加载模型文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "models/test/model_instruction.pth")
    state_dict = torch.load(filepath, map_location=common_device, weights_only=True)
    result = model.load_state_dict(state_dict)
    assert not result.missing_keys and not result.unexpected_keys, result
    device = common_device
    model.to(device)
    model.eval()

    torch.manual_seed(123)

    # 生成回复
    for i,entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)
            
        token_ids = generate(
            model,
            idx = text_to_token_ids(input_text, tokenizer),
            max_new_tokens=256,
            context_size=1024,
            eos_id = 50256)

        gen_texts = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            gen_texts[len(input_text):].replace("### Response:", "").strip()
        )
        test_data[i]["model_response"] = response_text
    
    # 保存文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "instruction/instruction-data-with-response.json")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(test_data, file, indent=4)

# ================================ 评估微调模型 ================================
# 测试ollama是否运行
def check_if_running(process_name):
    import psutil
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

def test_ollama_running():
    ollama_running = check_if_running("ollama")
    if not ollama_running:
        raise RuntimeError("Ollama not running.Launch ollama before proceeding.")
    
    print("Ollama running:", check_if_running("ollama"))

# 本地调用ollama
def query_model(
    prompt,
    model="qwen3:8b",
    url="http://localhost:11434/api/chat"):
    data = {
        "model":model,
        "messages":[{"role":"user", "content":prompt}],
        "think":False,
        "options":{"seed":123, "temperature":0, "num_ctx":2048}
    }

    payload = json.dumps(data).encode('utf-8')
    request = urllib.request.Request(
        url, data=payload, method="POST"
    )
    request.add_header("Content-Type", "application/json")

    response_data = ""
    with urllib.request.urlopen(request) as response:
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            if response_json.get("done"):
                break
            response_data += response_json["message"]["content"]
    
    return response_data

# 测试本地模型调用
def invoke_ollama():
    model = "qwen3:8b"
    result = query_model("What do llamas eat?", model)
    print(result)

# 测试之前的三个数据
def query_test_data():
    test_data = load_data("instruction/instruction-data-with-response.json")
    for entry in test_data[:3]:
        prompt = (
            f"Given the input '{format_input(entry)}' "
            f"and correct output '{entry['output']}', "
            f"score the model response '{entry['model_response']}'"
            f" on a scale from 0 to 100, where 100 is the best score."
        )
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry['model_response'])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n---------------------------------------------")

# 对所有输出打分
def generate_model_scores(json_data, json_key, model="qwen3:8b"):
    from tqdm import tqdm
    import re
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input '{format_input(entry)}' "
            f"and correct output '{entry['output']}', "
            f"score the model response '{entry[json_key]}'"
            f" on a scale from 0 to 100, where 100 is the best score."
            f"Respond with the integer number only."
        )
        score = query_model(prompt, model)
        match = re.search(r"\d+", score)
        if match:
            scores.append(int(match.group()))
        else:
            print(f"Could not convert score: {score}")
        # try:
        #     scores.append(int(score))
        # except ValueError:
        #     print(f"Could not convert score: {score}")
        #     continue

    return scores

# 测试模型打分
def test_generate_scores():
    test_data = load_data("instruction/instruction-data-with-response.json")
    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    if scores:
        print(f"Average score: {sum(scores)/len(scores):.2f}\n")

# practice7.4 LoRA微调
# 在台式机上做

# ================================ 指定执行 ================================
if __name__=="__main__":
    # load_data()
    # test_format_input()
    # prepare_data()
    # test_custom_collate()
    # test_instruction_loader()
    # load_pretrained_model()
    # val_simple_model()
    # calc_origin_loss()
    # train_instruction_ft()
    # compare_responses()
    # generate_response()
    # test_ollama_running()
    # invoke_ollama()
    # query_test_data()
    test_generate_scores()