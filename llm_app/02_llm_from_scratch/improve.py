import os
import time
import math
import torch
import tiktoken
import matplotlib.pyplot as plt
from model import GPTModel
from model import GPT_CONFIG_124M
from device import common_device
from embedding import create_dataloader_v1
from pre_train import calc_loss_batch, evaluate_model
from pre_train import text_to_token_ids, token_ids_to_text
from pre_train import generate_and_print_sample, plot_losses
from pre_train import generate_text_simple
from classifier import create_dataloader, load_gpt_model
from classifier import calc_accuracy_loader, train_classifier_simple, plot_values


# 初始化模型，创建数据加载器
def initial_trainnings():
    # 初始化模型
    model_config = GPT_CONFIG_124M.copy()
    model_config.update({"context_length": 256})
    device = common_device

    # 创建数据加载器
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "the-verdict.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        text_data = f.read()
    train_radio = 0.9
    split_idx = int(len(text_data) * train_radio)
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size = 2,
        max_length=model_config["context_length"],
        stride = model_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size = 2,
        max_length=model_config["context_length"],
        stride = model_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # 创建模型
    torch.manual_seed(123)
    model = GPTModel(model_config)
    model.to(device)
    model.eval()

    return model, train_loader, val_loader

# ================================ 学习率预热 ================================
# 提问 1：为什么需要预热学习率
def test_lr_warmup():
    model, train_loader, _ = initial_trainnings()

    # 训练参数设置
    num_epochs = 15
    initial_lr = 0.0001
    peak_lr = 0.01
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.2 * total_steps)

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1) 
    
    # test learning
    
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    global_step = -1
    track_lrs = []

    # 在训练循环中，关注学习率变化
    for epoch in range(num_epochs):
        for input,target in train_loader:
            optimizer.zero_grad()
            global_step += 1

            if global_step < warmup_steps:
                lr = initial_lr + lr_increment * global_step
            else:
                lr = peak_lr
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(optimizer.param_groups[0]["lr"])

    # 绘制学习率
    
    plt.ylabel("Learning rate")
    plt.xlabel("Step")
    total_training_steps = len(train_loader) * num_epochs
    plt.plot(range(total_training_steps), track_lrs)
    plt.show()

# ================================ 余弦衰减 ================================
# 提问 2：为什么需要余弦衰减
def test_cosine_decay():
    model, train_loader, _ = initial_trainnings()

    # 训练参数设置
    num_epochs = 15
    initial_lr = 0.0001
    peak_lr = 0.01
    min_lr = 0.1 * initial_lr
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.2 * total_steps)
    lr_increment = (peak_lr - initial_lr) / warmup_steps
    global_step = -1
    track_lrs = []

    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1) 

    # 训练
    for epoch in range(num_epochs):
        for input,target in train_loader:
            optimizer.zero_grad()
            global_step += 1

            if global_step < warmup_steps:
                lr = initial_lr + lr_increment * global_step
            else:
                process = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * process))
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(optimizer.param_groups[0]["lr"])

    # 绘制学习率
    plt.ylabel("Learning rate")
    plt.xlabel("Step")
    plt.plot(range(total_steps), track_lrs)
    plt.show()


# ================================ 梯度裁剪 ================================
# 提问 3：为什么需要裁剪梯度
# 找到最大梯度值
def find_highest_gradient(model:GPTModel):
    max_grad = None
    for param in model.parameters():
        if param.grad is not None:
            grad_values = param.grad.data.flatten()
            max_grad_param = grad_values.max()
            if max_grad is None or max_grad_param > max_grad:
                max_grad = max_grad_param
    return max_grad

def cal_gradient():
    model, train_loader, _ = initial_trainnings()
    model.train()
    input, target = next(iter(train_loader))
    loss = calc_loss_batch(input, target, model, common_device)
    loss.backward()
    print(find_highest_gradient(model))

    # 执行梯度裁剪后，观察最大梯度
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(find_highest_gradient(model))

# ================================ 修改训练函数 ================================
def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer, 
                warmup_steps, initial_lr=3e-5, min_lr=1e-6):
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []
    tokens_seen, global_step = 0, -1

    peak_lr = optimizer.param_groups[0]["lr"]
    total_steps = len(train_loader) * n_epochs
    lr_increment = (peak_lr - initial_lr) / warmup_steps

    # 开启训练循环
    for epoch in range(n_epochs):
        model.train()
        for input, target in train_loader:
            optimizer.zero_grad()
            global_step += 1

            # 调整学习率
            if global_step < warmup_steps:
                lr = initial_lr + lr_increment * global_step
            else:
                progress = (global_step - warmup_steps) / (total_steps - warmup_steps)
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            track_lrs.append(lr)
            
            # 计算损失
            loss = calc_loss_batch(input, target, model, device)
            loss.backward()
            # 梯度裁剪
            if global_step >= warmup_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            tokens_seen += input.numel()

            # 验证
            # 以一定频次对模型当前训练水平做验证
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")

        # 每个轮次打印生成文本
        generate_and_print_sample(model, tokenizer, device, start_context)
    
    return train_losses, val_losses, track_tokens_seen, track_lrs

# 开始训练
def begin_train():
    model, train_loader, val_loader = initial_trainnings()

    peak_lr = 5e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = common_device

    n_epochs = 15
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.2 * total_steps)
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device,
        n_epochs, eval_freq=5, eval_iter=1, start_context="Every effort moves you",
        tokenizer=tokenizer, warmup_steps=warmup_steps, initial_lr=1e-5, min_lr=1e-5
    )

    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # 绘制学习率曲线
    plt.figure()
    plt.ylabel("Learning rate")
    plt.xlabel("Step")
    plt.plot(range(total_steps), lrs)
    plt.show()

# ================================ LoRA ================================
# 提问 4： 简述LoRA策略
# 定义LoRA层
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha) -> None:
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
    
    def forward(self, x):
        return self.alpha * (x @ self.A @ self.B)

# 定义替换层
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha) -> None:
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

# 替换函数：将Linear替换为Linear-with-LoRA
def replace_linear_with_lora(model:GPTModel, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            replace_linear_with_lora(module, rank, alpha)

# 对分类任务进行LoRA高效微调
def test_LoRA():
    # 准备分类数据加载器
    train_loader,val_loader, test_loader = create_dataloader()
    print("Train loader:")
    for input,target in train_loader:
        pass
    print("Input batch dimensions:", input.shape)
    print("Label batch dimensions:", target.shape)
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    # 加载模型
    model = load_gpt_model()

    tokenizer = tiktoken.get_encoding("gpt2")

    # 测试能否生成流利文本
    text_1 = "Every effort moves you"
    token_ids = generate_text_simple(
        model,
        idx = text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size = GPT_CONFIG_124M["context_length"]
    ) 

    print(token_ids_to_text(token_ids, tokenizer))

    # 微调分类模型，替换输出头
    torch.manual_seed(123)
    num_classes = 2
    model.out_head = torch.nn.Linear(GPT_CONFIG_124M["emb_dim"], out_features=num_classes)
    device = common_device
    model.to(device)

    # 计算初始分类准确率
    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

    # 冻结原始模型参数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters before: {total_params:,}")
    for param in model.parameters():
        param.requires_grad = False
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters after: {total_params:,}")

    # 使用LoRA替换linear
    replace_linear_with_lora(model, rank = 16, alpha =16)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable LoRA parameters: {total_params:,}")

    # 打印模型
    model.to(device)
    print(model)

    # 计算初始分类准确率
    torch.manual_seed(123)
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
    print(f"Training accuracy before LoRA: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy before LoRA: {val_accuracy*100:.2f}%")
    print(f"Test accuracy before LoRA: {test_accuracy*100:.2f}%")

    # 使用LoRA微调
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = \
        train_classifier_simple(model, train_loader, val_loader, optimizer, device,
            num_epochs, eval_freq=50, eval_iter=5)
    end_time = time.time()
    excution_time_minutes = (end_time - start_time) / 60

    print(f"Training completed in {excution_time_minutes:.2f} minutes.")

    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses, label="loss")
    
    # 计算准确率
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
    print(f"Training accuracy after LoRA: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy after LoRA: {val_accuracy*100:.2f}%")
    print(f"Test accuracy after LoRA: {test_accuracy*100:.2f}%")

# ================================ 推理模型思考 ================================
# 提问 5：推理模型的使用场景与优劣点
# 提问 6：构建与优化推理模型的四大方法

# ================================ 指定执行 ================================
if __name__=="__main__":
    # test_lr_warmup()
    # test_cosine_decay()
    # cal_gradient()
    # begin_train()
    test_LoRA()