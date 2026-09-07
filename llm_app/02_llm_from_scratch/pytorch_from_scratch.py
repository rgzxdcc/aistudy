import torch

# =======================torch环境=======================

# 测试torch环境是否安装
print(torch.__version__)
# 测试torch安装版本是否支持mps加速
print(torch.backends.mps.is_available())

# =======================创建张量=======================

# 体会张量维度、数据类型
tensor0d = torch.tensor(1)
tensor1d = torch.tensor([1.0, 2.0])
tensor2d = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
tensor3d = torch.tensor([[[1, 2], [3, 4]],
                        [[5, 6], [7, 8]]])
print(tensor0d.shape)
print(tensor1d.shape)
print(tensor2d.shape)
print(tensor3d.shape)
print(tensor0d.dtype) # torch.int64
print(tensor1d.dtype) # torch.float32

# 更改张量类型
float_scalar = tensor0d.to(torch.float32)
print(float_scalar.dtype) # torch.int64 -> torch.float32

# =======================常见张量操作=======================
tensor_oper = torch.tensor([[1, 2, 3],
                            [4, 5, 6]])
print(tensor_oper)
print(tensor_oper.shape)
print(tensor_oper.T)

tensor_oper = tensor_oper.view(3, 2)
print(tensor_oper)
print(tensor_oper.shape)
print(tensor_oper.T)

# 矩阵乘法
print(tensor_oper @ tensor_oper.T)

# =======================autograd=======================
from torch._decomp.decompositions import tensor_split_tensor_indices_or_sections_py_impl
from torch.nn import parameter
import torch.nn.functional as F
y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
w1 = torch.tensor([2.2], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
z = x1 * w1 + b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

from torch.autograd import grad

# 手动调用grad，计算梯度
grad_w1 = grad(loss, w1, retain_graph=True)
grad_b = grad(loss, b, retain_graph=True)

print(grad_w1)
print(grad_b)

# 自动计算
loss.backward()
print(w1.grad)
print(b.grad)

# =======================multilayer precetron=======================
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(
            
            # 第一个隐藏层 
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 第二个隐藏层
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # 输出层
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers.forward(x)
        return logits


# 创建模型
torch.manual_seed(123)
model = NeuralNetwork(50, 3)
print(model)

# 统计参数量
param_num = sum( p.numel() for p in model.parameters() if p.requires_grad)
print(f"total number of trainable model parameters: {param_num}")

print(model.layers[0].weight)
print(model.layers[0].bias)
# 构造函数 in 在前，参数形状 out 在前
print(model.layers[0].weight.shape) # torch.Size([30, 50])
print(model.layers[0].bias.shape)

# 根据输入计算输出
torch.manual_seed(123)
X = torch.rand((1, 50))
with torch.no_grad():
    a = model(X)
    out = torch.softmax(a, dim=1)
print(out)


# =======================dataloader=======================
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])
X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6]
])
y_test = torch.tensor([0, 1])

# 自定义数据集
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y) -> None:
        self.features = X
        self.labels = y

    # 检索一条数据记录及其对应标签
    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    # 返回数据集总长度
    def __len__(self):
        return self.labels.shape[0]

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

print(len(train_ds))
print(len(test_ds))

# 创建数据加载器
from torch.utils.data import DataLoader

torch.manual_seed(123)

train_loader = DataLoader(
    dataset = train_ds, 
    batch_size = 2, 
    shuffle = True, 
    num_workers=0,
    drop_last=True
)
test_loader = DataLoader(
    dataset = test_ds, 
    batch_size = 2, 
    shuffle = False, 
    num_workers=0
)

# 迭代数据加载器
for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx + 1}: ", x, y)


# =======================train=======================
import torch.nn.functional as F

torch.manual_seed(123)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

# 总轮次：一轮模型学习不到东西，需要多轮次反复学习
num_epoches = 3
for epoch in range(num_epoches):
    
    # 开启模型训练模式
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        # 将上一轮梯度置为0，防止梯度累积，执行反向传播计算梯度，优化器使用梯度更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epoches:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")

    # 开启模型评估模式
    model.eval()
    # 模型评估代码...


# 计算神经网络参数
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total model parameters number: {num_params}")

# 模型预测
model.eval()
with torch.no_grad():
    out = model(X_train)
print(out)

# 转换为概率
torch.set_printoptions(sci_mode=False)
probas = torch.softmax(out, dim=1)
print(probas)

# 转换为最大序列号：可以跳过生成概率，直接对得分做argmax
predictions = torch.argmax(probas, dim=1)
predictions = torch.argmax(out, dim=1)
print(predictions)

# 对比预测结果与标签
print(predictions == y_train)

# 计算正确预测数量，以及准确率
correct_num = torch.sum(predictions == y_train)
print(f"correct predicitons number: {correct_num}")
accuracy = correct_num / len(X_train)
print(f"predict accuracy: {accuracy}")

# 封装计算预测准确率函数
def compute_accuracy(model, dataloader):
    model.eval()

    corrects = 0
    total_examples = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        with torch.no_grad():
            logits = model(features)
            predictions = torch.argmax(logits, dim=1)

            corrects += sum(predictions == labels)
            total_examples += len(features)

    return (corrects / total_examples).item()

# 计算训练集、测试集准确率
accu_train = compute_accuracy(model, train_loader)
accu_test = compute_accuracy(model, test_loader)
print(f"accuracy of train : {accu_train}")
print(f"accuracy of test : {accu_test}")

# =======================model save and load=======================
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "pytorch_test.pth")
torch.save(model.state_dict(), model_path)
print(model.state_dict())
model2 = NeuralNetwork(2, 2)
model2.load_state_dict(torch.load(model_path))
accu_train = compute_accuracy(model2, train_loader)
accu_test = compute_accuracy(model2, test_loader)
print(f"accuracy for train of model2 : {accu_train}")
print(f"accuracy for test of model2 : {accu_test}")


# =======================GPU优化=======================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# 改动两个地方：model以及训练、测试数据转移到device上
tensor_1 = torch.tensor([1, 2, 3])
tensor_2 = torch.tensor([4, 5, 6])
print(tensor_1 + tensor_2)
# 移到GPU上
tensor_1 = tensor_1.to(device)
tensor_2 = tensor_2.to(device)
print(tensor_1 + tensor_2)

## practice A.4: 寻找 GPU 反超 CPU 的矩阵规模临界点
import timeit
# timeit.timeit(lambda: t1 @ t2.T, number=1000)
dim = 10
cpu_time = 0
gpu_time = 1
# 设置GPU阻塞等待，让GPU真正算完完成计时
def gpu_matmul(a, b):
    out = a @ b
    torch.mps.synchronize()
    return out

while(cpu_time < gpu_time):
    t1 = torch.rand(dim, dim)
    t2 = torch.rand(dim, dim)
    cpu_time = timeit.timeit(lambda: t1 @ t2, number = 1000)

    t1_gpu = t1.to(device)
    t2_gpu = t2.to(device)
    
    # GPU预热：排除GPU首次调用的初始化开销
    _ = t1_gpu @ t2_gpu
    torch.mps.synchronize()
    gpu_time = timeit.timeit(lambda: gpu_matmul(t1_gpu, t2_gpu), number = 1000)

    dim += 1

print(f"临界点：矩阵维度为{dim - 1} *{dim - 1}, 超过该规模，GPU开始变更快")



