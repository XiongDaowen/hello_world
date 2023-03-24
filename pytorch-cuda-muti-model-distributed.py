import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time

# 打印 GPU 数量和型号信息
device_count = torch.cuda.device_count()
print(f"Using {device_count} GPUs!")
for i in range(device_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
 

# 初始化进程组
dist.init_process_group(
    backend='tcp',  # 使用tcp作为后端通信
    init_method='tcp://localhost:23456',  # 指定服务器地址
    rank=0,  # 当前进程的rank
    world_size=8  # 进程组中的总进程数
)

# 定义模型
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 8 * 8, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 定义数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载 CIFAR-10 数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, sampler=train_sampler)

# 创建模型实例并将其分配到多个 GPU 上
print("创建模型实例并将其分配到多个 GPU 上")
model = Model()
model = nn.DataParallel(model)
model.cuda()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()
# 训练模型
for epoch in range(3):
    running_loss = 0.0
    train_sampler.set_epoch(epoch)
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # 每迭代 100 个 batch 打印一次 loss
        if i % 100 == 99:
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
print('Finished Training')

end_time = time.time()
print(f"Training time: {end_time - start_time:.2f} seconds")

# 加载 CIFAR-10 测试集
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

# 在测试集上进行验证
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on test set: %.2f %%' % (100 * correct / total))
