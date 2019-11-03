import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
import tensorflow as tf

class ConvNet(nn.Module):

    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
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


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def trainCNN(self, net, x, y):
        x = tf.Tensor(x)
        y = tf.Tensor(y)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            # With a batch size of 4 in each iteration
            for i, data in zip(x,y):  # trainloader reads data using torchvision
                inputs, labels = data
                inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                outputs = net(inputs)
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

    def testCNN(self, net, x, y):
        x = tf.Tensor(x)
        y = tf.Tensor(y)
        correct = 0
        total = 0
        for data in zip(x,y):
            images, labels = data
            outputs = net(Variable(images))
            _, predicted = torch.max(outputs.data, 1)  # Find the class index with the maximum value.
            total += labels.size(0)
            correct += (predicted == labels).sum()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))

