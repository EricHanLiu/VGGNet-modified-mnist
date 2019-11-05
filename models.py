import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable

PATH = './cifar_net.pth'


class CNN(nn.Module):
    def __init__(self, x, y, num_classes=10):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        print("Layer 1 completed!")
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        print("Layer 2 completed!")
        self.fc = nn.Linear(7 * 7 * 32, num_classes)
        self.x = x
        self.y = y

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def train_cnn(self):

        tensor_x = torch.tensor(self.x)
        tensor_y = torch.tensor(self.y)
        my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset
        train_loader = torch.utils.data.DataLoader(my_dataset, batch_size=4)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            # With a batch size of 4 in each iteration
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)
                inputs = inputs.unsqueeze(1)
                inputs = inputs.view(1, -1)

                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        torch.save(self.state_dict(), PATH)
        print('Training saved')

    def test_cnn(self):
        tensor_x = torch.stack([torch.Tensor(i) for i in self.x])
        tensor_y = torch.stack([torch.Tensor(i) for i in self.y])
        my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset
        correct = 0
        total = 0
        test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=4)
        for data in test_loader:
            images, labels = data
            outputs = self(Variable(images))
            _, predicted = torch.max(outputs.data, 1)  # Find the class index with the maximum value.
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
