import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y


# 定义网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 准备数据
data = torch.randn(100, 10)
target = torch.randint(0, 2, (100,))
dataset = MyDataset(data, target)
dataloader = DataLoader(dataset, batch_size=10)

# 定义模型、损失函数、优化器
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
#打印模型
print(model)

# 训练模型
for epoch in range(10):
    # 打印epoch
    print('epoch {}'.format(epoch + 1))
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# 测试模型
test_data = torch.randn(20, 10)
test_target = torch.randint(0, 2, (20,))
test_dataset = MyDataset(test_data, test_target)
test_dataloader = DataLoader(test_dataset, batch_size=5)
total = 0
correct = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # 打印预测结果
        print('predicted: ', predicted)
        print('labels: ', labels)

print('Accuracy: %.2f %%' % (100 * correct / total))
