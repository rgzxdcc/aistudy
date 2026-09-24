from json import load
import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from pandas import DataFrame
import tiktoken
from transformers.utils.type_validators import label_to_id_validation
from gpt_download import download_and_load_gpt2
from model import GPTModel
from pre_train import common_device, load_weights_into_gpt
    

# ================================ 准备数据集 ================================
def prepare_datasets():

    # 读取原始数据集
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, "sms_spam_collection/SMSSpamCollection.tsv")
    df = pd.read_csv(file_name, sep="\t", header=None, names = ["Label", "Text"])
    print(df)

    # 观察类别分布
    print(df["Label"].value_counts())

    # 提问 1：为什么需要下采样 
    # 下采样
    def create_balanced_dataset(df: DataFrame):
        spam_num = df[df["Label"] == "spam"].shape[0]
        ham_subset = df[df["Label"] == "ham"].sample(spam_num, random_state=123)
        balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
        return balanced_df

    # 执行下采样
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    # 将标签转为0 1
    balanced_df["Label"] = balanced_df["Label"].map({"ham":0, "spam":1})
    print(balanced_df["Label"].value_counts())

    # 划分数据集
    def random_split(df:DataFrame, train_frac, val_frac):
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        train_end = int(train_frac * len(df))
        val_end = train_end + int(val_frac * len(df))

        train_df = df[:train_end]
        val_df = df[train_end:val_end]
        test_df=df[val_end:]

        return train_df, val_df, test_df
    
    # 执行数据集划分
    train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)

    # 保存数据集
    train_df.to_csv(os.path.join(script_dir, "sms_spam_collection/train.csv"), index=False)
    val_df.to_csv(os.path.join(script_dir, "sms_spam_collection/validation.csv"), index=False)
    test_df.to_csv(os.path.join(script_dir, "sms_spam_collection/test.csv"), index=False)

# ================================ 创建数据加载器 ================================
def test_data():
    tokenizer = tiktoken.get_encoding("gpt2")
    token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    print(token_id)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_name = os.path.join(script_dir, "sms_spam_collection/train.csv")
    
    data = pd.read_csv(file_name)
    encoded_texts = [tokenizer.encode(text) for text in data["Text"]]
    print(len(encoded_texts))
    [print(text) for text in data["Text"]]

# 提问 2：为什么要做数据填充对齐
# 提问 3：此处的Dataset与之前训练大模型定义的Dataset结构有什么不同
# 创建数据集：填充序列对齐
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        # 确定最大文本长度
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
        self.encoded_texts = [text[:self.max_length] for text in self.encoded_texts]
        
        # 填充文本
        self.encoded_texts = [text + [pad_token_id] * (self.max_length - len(text)) 
            for text in self.encoded_texts]
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (torch.tensor(encoded, dtype=torch.long),
                torch.tensor(label, dtype=torch.long))
    
    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            length = len(encoded_text)
            if length > max_length:
                max_length = length
        return max_length

# 测试数据集
def create_dataloader():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, "sms_spam_collection/train.csv")
    val_file = os.path.join(script_dir, "sms_spam_collection/validation.csv")
    test_file = os.path.join(script_dir, "sms_spam_collection/test.csv")

    tokenizer = tiktoken.get_encoding("gpt2")

    # 分别构建训练、验证、测试三个数据集
    train_dataset = SpamDataset(
        csv_file = train_file,
        tokenizer = tokenizer,
        max_length=None
    )
    val_dataset = SpamDataset(
        csv_file=val_file,
        tokenizer =tokenizer,
        max_length = train_dataset.max_length
    )
    test_dataset = SpamDataset(
        csv_file=test_file,
        tokenizer =tokenizer,
        max_length = train_dataset.max_length
    )

    # 分别创建数据加载器
    from torch.utils.data import DataLoader

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        drop_last = False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = batch_size,
        drop_last = False,
        num_workers=num_workers
    )
    
    # # 打印每个批次维度
    # for input,label in train_loader:
    #     pass
    # print("input of train_loader shape: ", input.shape)
    # print("label of train_loader shape: ", label.shape)

    # # 分别打印所有批次数
    # print(f"{len(train_loader)} train batches")
    # print(f"{len(val_loader)} validation batches")
    # print(f"{len(test_loader)} test batches")

    return train_loader,val_loader, test_loader


# practice6.1
# 将SpamDataset的max_length传入1024，此处性能开销变大

# ================================ 初始化模型 ================================
# 加载预训练模型
def load_gpt_model():

    # 组织模型参数配置
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    from pre_train import model_configs
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 读取并加载模型权重
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir=os.path.join(script_dir, "models")
    )
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    return model

# 测试模型能否顺利生成文本
def test_model_load():
    from pre_train import generate_text_simple
    from pre_train import text_to_token_ids, token_ids_to_text

    model = load_gpt_model()

    # 测试顺否能够生成文本
    tokenizer = tiktoken.get_encoding("gpt2")
    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(
        model = model,
        idx = text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size = model.pos_emb.weight.shape[0]
    )
    print(token_ids_to_text(token_ids, tokenizer))

    # 测试模型是否具备文本分类能力
    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )
    token_ids = generate_text_simple(
        model = model,
        idx = text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size = model.pos_emb.weight.shape[0]
    )
    print(token_ids_to_text(token_ids, tokenizer))


# ================================ 添加分类头 ================================
# 提问 4：为什么可以微调几个层，不是微调所有层
# 为实现二分类任务，改造GPTModel
def model_modify():

    # 加载预训练模型 
    model = load_gpt_model()

    # 冻结模型
    for param in model.parameters():
        param.requires_grad = False

    # 替换输出层
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        model.tok_emb.weight.shape[1],
        num_classes
    )

    # 将最终层归一化与最后一个Transformer块的梯度激活
    for param in model.final_norm.parameters():
        param.requires_grad = True
    for param in model.tfb[-1].parameters():
        param.requires_grad = True

    # # 查看修改后模型输出
    # tokenizer = tiktoken.get_encoding("gpt2")
    # inputs = tokenizer.encode("Do you have time")
    # inputs = torch.tensor(inputs).unsqueeze(0)
    # print("inputs: ", inputs)
    # print("inputs dimensions: ", inputs.shape)
    # with torch.no_grad():
    #    logits = model(inputs)
    # print("logits: ", logits)
    # print("logits dimensions: ", logits.shape)

    # # 仅关注最后一个词元
    # print("Last output token: ", logits[:, -1, :])
    # # 输出->概率分数->对应标签
    # probas = torch.softmax(logits[:, -1, :], dim=-1)
    # label = torch.argmax(probas)
    # print("Class label:", label.item())

    return model

# 提问 5：为什么只对最后一个词元感兴趣
# ================================ 计算损失和准确率 ================================

# 计算数据加载器准确率
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += ((predicted_labels == target_batch).sum().item())
        else:
            break
    return correct_predictions / num_examples

# 测试当前模型对个数据集的分类准确率
def test_origin_accuracy():
    device = common_device
    model = model_modify()
    model.to(device)

    train_loader, val_loader, test_loader = create_dataloader()

    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# 计算单批次损失
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss  = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# 计算多批次平均损失
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches

# 计算每个数据集损失
def test_origin_loss():
    device = common_device
    model = model_modify()
    model.to(device)

    train_loader, val_loader, test_loader = create_dataloader()

    torch.manual_seed(123)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

# practice6.2
def practice6_2():
    # 加载预训练模型 
    model = load_gpt_model()

    # 不执行模型冻结

    # 替换输出层
    num_classes = 2
    model.out_head = torch.nn.Linear(
        model.tok_emb.weight.shape[1],
        num_classes
    )

    # 正常训练

# practice6.3：比较微调第一个词元与最后一个词元
# 微调第一个词元类似抢答，拿到的上下文信息不全，就开始回答。对大模型来说，类似胡说八道

# ================================ 微调模型 ================================
def train_classifier_simple(model, train_loader, val_loader, optimizer, 
                                device, num_epoches, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epoches):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_batch.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
            
        # 为每一轮计算准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}% | ")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen

# 评估模型（计算数据集损失）
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

# 开启微调训练
def train_fine_tuning():
    import time

    # 加载预训练模型
    device = common_device
    model = model_modify()
    model.to(device)

    train_loader, val_loader, test_loader = create_dataloader()

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 5

    # 进行微调训练
    train_losses, val_losses, train_accs, val_accs, examples_seen = \
        train_classifier_simple(model, train_loader, val_loader, optimizer,
            device, num_epochs, eval_freq=50, eval_iter=5)

    # 打印训练总用时
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 保存模型微调权重
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "models/test/model_classifier.pth")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)

    # 绘制损失、准确率曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")

    # 测试所有数据准确率
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Training accuracy : {train_accuracy*100:.2f}%")
    print(f"Validation accuracy : {val_accuracy*100:.2f}%")
    print(f"Test accuracy : {test_accuracy*100:.2f}%")

# 绘制分类损失曲线
def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.show()


# ================================ 使用分类模型 ================================
# 分类函数
def classify_review(text, model, tokenizer, device,
                    max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]

    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

def test_classifier():
    # 加载模型
    CHOOSE_MODEL = "gpt2-small (124M)"
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }
    from pre_train import model_configs
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 读取并加载模型权重
    model = GPTModel(BASE_CONFIG)
    # 替换输出层
    num_classes = 2
    model.out_head = torch.nn.Linear(
        model.tok_emb.weight.shape[1],
        num_classes
    )

    # 加载模型文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, "models/test/model_classifier.pth")
    state_dict = torch.load(filepath, map_location=common_device, weights_only=True)
    result = model.load_state_dict(state_dict)
    assert not result.missing_keys and not result.unexpected_keys, result
    device = common_device
    model.to(device)
    model.eval()

    # 对文本进行分类
    tokenizer = tiktoken.get_encoding("gpt2")
    
    text_1 = ("You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.")
    print(classify_review(text_1, model, tokenizer, device, max_length=120))
    text_2 = ("Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!")

    print(classify_review(text_2, model, tokenizer, device, max_length=120))
    

# ================================ 指定执行 ================================
if __name__=="__main__":
    # prepare_datasets()
    # test_data()
    # create_dataloader()
    # load_gpt_model()
    # test_model_load()
    # model_modify()
    # test_origin_accuracy()
    # test_origin_loss()
    # train_fine_tuning()
    test_classifier()