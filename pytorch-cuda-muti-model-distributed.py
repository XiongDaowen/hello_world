import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os

os.environ['RANK'] = '0'  # 将当前进程的rank设置为0

batch_size = 64
learning_rate = 0.01
num_epochs = 10
world_size = 8
dist_backend = "nccl" # or "gloo"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.softmax(self.fc3(x), dim=1)
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# initialize the process group
dist.init_process_group(backend=dist_backend)

# get the rank of the current process and the total number of processes
rank = dist.get_rank()
size = dist.get_world_size()

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    sampler=train_sampler
)

# set the device for this process
device = torch.device("cuda:{}".format(rank))

model = Net()

# if there are multiple GPUs, use DistributedDataParallel to distribute the model across them
if torch.cuda.device_count() > 1:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# start the timer
start_time = time.time()

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, num_epochs, batch_idx, len(train_loader), loss.item()))

# stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# synchronize all processes and sum the elapsed time across all GPUs
dist.barrier()
total_time = torch.tensor([elapsed_time]).cuda()
dist.all_reduce(total_time)
elapsed_time = total_time.item() / world_size

# evaluate the model
model.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    print('Accuracy of the model on the {} train images: {} %'.format(len(train_dataset), 100 * correct / total))
# print the elapsed time
print("Training completed in {:.2f} seconds".format(elapsed_time))
