import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

batch_size = 64
learning_rate = 0.01
num_epochs = 10

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

# create multiple datasets, one for each GPU
train_datasets = [torch.utils.data.Subset(train_dataset, indices=list(range(i, len(train_dataset), torch.cuda.device_count()))).shuffle(True) for i in range(torch.cuda.device_count())]

# create multiple data loaders, one for each GPU
train_loaders = [DataLoader(dataset=train_datasets[i], batch_size=batch_size, shuffle=True) for i in range(torch.cuda.device_count())]

# check how many GPUs are available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using", torch.cuda.device_count(), "GPUs")
else:
    device = torch.device("cpu")
    print("Using CPU")

model = Net().to(device)

# if there are multiple GPUs, use DataParallel to distribute the data across them
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# start the timer
start_time = time.time()

for epoch in range(num_epochs):
    for loader_idx, train_loader in enumerate(train_loaders):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Epoch [{}/{}], Loader [{}/{}], Batch [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, num_epochs, loader_idx+1, len(train_loaders), batch_idx, len(train_loader), loss.item()))

# stop the timer and calculate the elapsed time
end_time = time.time()
elapsed_time = end_time - start_time

# evaluate the model
model.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for data, target in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    if total != 0:
        print('Accuracy of the model on the {} train images: {} %'.format(len(train_dataset), 100 * correct / total))
    else:
        print('Error: total number of samples processed is zero')

# print the elapsed time
print("Training completed in {:.2f} seconds".format(elapsed_time))
