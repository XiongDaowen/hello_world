import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data.distributed
import torch.nn.parallel.distributed as dist

import time

单机多卡训练不一定比单级单卡训练快，这取决于多卡训练的实现方式和具体应用场景。一般情况下，单机多卡训练可以利用多个 GPU 的并行计算能力，提高训练速度，但也存在一些限制，例如 GPU 之间的通信、内存的限制等问题。

具体来说，单机多卡训练的优势主要体现在以下几个方面：

并行计算能力：多个 GPU 可以同时进行计算，可以加快训练速度。
内存利用率：单机多卡训练可以将模型和数据分配到多个 GPU 上，充分利用 GPU 内存，减少 GPU 的内存压力。
灵活性：单机多卡训练可以根据实际情况灵活调整模型和数据的分配方式，以达到最优的训练效果。

模型大小：一般来说，单机多卡训练适用于模型参数量在几百万到数千万之间的任务，例如 ResNet-50、VGG-16 等经典的卷积神经网络模型。当模型参数量超过数千万时，可能需要采用更高级别的分布式训练方式，例如多机多卡训练。
数据大小：一般来说，单机多卡训练适用于数据集大小在数百 GB 到数 TB 之间的任务。当数据集大小超过数 TB 时，可能需要采用分布式存储和分布式训练的方式。

***********************

在多机多卡训练时，模型和数据的分布机制通常需要根据具体的分布式训练框架进行调整。不同的框架可能有不同的实现方式，但是它们通常都需要考虑以下两个问题：

模型的分布机制：在分布式训练中，模型通常需要被分布到多个机器和多个 GPU 上进行计算。常见的方式是将模型的不同部分分布到不同的机器或 GPU 上，例如将
不同的层分布到不同的机器或 GPU 上，或者将每个 batch 的不同样本分布到不同的机器或 GPU 上。分布式训练框架通常需要考虑如何实现这些分布方式，并保证
模型的正确性和一致性。

数据的分布机制：在分布式训练中，数据也需要被分布到多个机器和多个 GPU 上进行计算。常见的方式是将不同的样本或 batch 分布到不同的机器或 GPU 上，
或者将每个样本或 batch 的不同部分分布到不同的机器或 GPU 上。分布式训练框架通常需要考虑如何实现这些分布方式，并保证数据的正确性和一致性

也就是说
多机多卡和单机多卡的主要区别在于模型和数据的分布方式。在单机多卡训练中，模型和数据都是分布在同一台机器的多个 GPU 上进行计算。而在多机多卡训练中，模型和数据需要被分布到多台机器和多个 GPU 上进行计算。

***********************

关键代码：

# 使用 torch.distributed.launch 脚本启动了 4 个进程，每个进程运行一份代码，对应 4 台机器。
# 设置分布式环境参数
world_size = 4  # 使用 4 台机器进行训练
rank = 0  # 当前机器的 rank，取值范围为 0 到 world_size-1

#在多机多卡训练中，每台机器的 rank 都是不同的。rank 参数指定了当前进程在分布式训练中的编号，范围从 0 到 world_size-1，其中 world_size 是指参与分布式训练的总进程数。
#每个进程都需要有自己的 rank 编号，以便在分布式训练中进行通信和同步。在实际使用中，可以将每个进程的 rank 作为启动参数传递给程序，或者使用环境变量来指定。

#在多机多卡训练中，每台机器上的代码是基本一致的，但是每台机器上运行的代码可能需要做一些微调，以便适应不同的环境和参数。
#具体来说，每台机器上需要指定当前机器的 rank 和总的 world_size，以及启动分布式训练环境。
#在 PyTorch 中，可以使用 torch.distributed.init_process_group() 函数来初始化分布式训练环境，以便不同进程之间进行通信和同步。

#在多机多卡训练中，NCCL 通信的分配由 PyTorch 的分布式训练模块自动完成。
#PyTorch 提供了多种分布式训练后端，包括 NCCL、Gloo 和 MPI 等，可以根据实际情况选择不同的后端。
#对于 NCCL 后端，PyTorch 会自动根据硬件和网络配置来选择最优的通信方式和通信协议，以提高训练效率。
#通常情况下，NCCL 会优先使用 RDMA 方式进行通信，以避免网络传输的瓶颈和额外的 CPU 开销。


# 初始化分布式环境
# 在训练过程中，使用 nn.parallel.DistributedDataParallel 来实现模型的分布式计算和通信，以及在多个进程之间进行数据同步。
# 每台机器上的 init_method 参数应该是相同的，以确保进程之间可以相互通信和同步。
dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', world_size=world_size, rank=rank)

# 在当前机器上只使用 GPU 0 进行训练
device = torch.device(f'cuda:{rank}')

# 创建模型实例并将其分配到 GPU 上
model = Model().to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

# 要使用这台机器的所有显卡
device = torch.device(f'cuda:{rank}')
device_ids = list(range(torch.cuda.device_count()))
model = Model().to(device)
model = nn.parallel.DistributedDataParallel(model, device_ids=device_ids)

# 在数据加载时，使用了 torch.utils.data.distributed.DistributedSampler 和 torch.utils.data.DataLoader 来进行分布式数据加载。
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, sampler=train_sampler)

 # 在多个进程中启动训练
mp.spawn(train, args=(world_size,), nprocs=world_size)

# 启动多个机器
# 启动分布式训练的命令通常是在其中一台机器上运行的。具体来说，可以将其中一台机器作为“主节点”，在该节点上运行启动命令，并指定其他机器的 IP 地址和端口号。
# 在多机多卡训练中，所有机器先启动一个分布式训练环境，以便在不同节点之间进行通信和同步。在分布式训练环境初始化完成后，主节点才能够向其他节点下达命令。
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=8888 your_train_script.py --arg1 --arg2 ...

#大模型训练的技术和方法
数据并行：数据并行是将模型分散到多个 GPU 或多台机器上，每个 GPU 或机器负责计算一部分数据，并将结果汇总到一起。数据并行可以有效地加速模型的训练，但需要考虑通信和同步的开销。
模型并行：模型并行是将模型分解成多个部分，并将不同的部分分散到不同的 GPU 或机器上进行计算。模型并行可以用于处理参数量很大的模型，但需要考虑模型分解的方式和数据划分的问题。
混合并行：混合并行是将数据并行和模型并行结合起来，同时使用多个 GPU 或机器进行计算。混合并行可以充分利用多个 GPU 或机器的计算能力，但需要解决通信和同步的问题。
梯度累积：梯度累积是将多个小批量数据的梯度进行累积，形成一个大批量数据的梯度，然后再进行参数更新。梯度累积可以用于处理显存不足的情况，但需要考虑训练时间的问题。
梯度压缩：梯度压缩是将梯度进行压缩，以减少通信和同步的开销。梯度压缩可以用于处理大模型分布式训练中的通信和同步问题，但需要考虑压缩算法的效率和精度。

#超大模型
Horovod：Horovod 是一个由 Uber 开源的分布式深度学习框架，专门用于超大模型的训练。Horovod 支持多种分布式训练的方式，包括数据并行、模型并行、混合并行等，支持多机多卡、单机多卡等多种部署方式。
Megatron：Megatron 是 NVIDIA 开源的超大规模语言模型训练框架，专门用于训练超大模型。Megatron 使用模型并行的方式，将模型分解成多个部分，分散到多个 GPU 或机器上进行计算。
DeepSpeed：DeepSpeed 是微软开源的分布式深度学习框架，专门用于训练超大模型。DeepSpeed 提供了多种分布式训练的方式，包括数据并行、模型并行、混合并行等，支持多机多卡、单机多卡等多种部署方式。

###########################################################################
# 打印 GPU 数量和型号信息
device_count = torch.cuda.device_count()
print(f"Using {device_count} GPUs!")
for i in range(device_count):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# 初始化分布式训练环境
torch.distributed.init_process_group(backend='nccl', init_method='env://')

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

# 使用 DistributedSampler 对数据集进行拆分
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4, sampler=train_sampler)

# 创建模型实例并将其分配到多个 GPU 上
model = Model().to('cuda')
model = nn.parallel.DistributedDataParallel(model)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

start_time = time.time()
# 训练模型
for epoch in range(3):
    running_loss = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
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
