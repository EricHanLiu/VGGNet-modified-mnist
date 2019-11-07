import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
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

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torch.utils.data.TensorDataset(torch.tensor(self.x), torch.tensor(self.y))
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

        classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        torch.save(self.state_dict(), PATH)
        print('Training saved')

    def test_cnn(self):
        testset = torch.utils.data.TensorDataset(torch.tensor(self.x), torch.tensor(self.y))
        testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
