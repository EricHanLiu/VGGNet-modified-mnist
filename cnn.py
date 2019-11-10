import numpy as np  # to handle matrix and data operation
import pandas as pd  # to read csv and handle dataframe
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.autograd import Variable
from sklearn.model_selection import train_test_split

# constants
EPOCHS = 4
BATCH_SIZE = 32

# input and target data
X = pd.read_pickle('train_max_x')[:1000]
y = np.array(pd.read_csv('train_max_y.csv'))[:1000, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

torch_X_train = torch.from_numpy(X_train).type(torch.LongTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)  # data type is long

# create feature and targets tensor for test set.
torch_X_test = torch.from_numpy(X_test).type(torch.LongTensor)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)  # data type is long

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)


def fit(model, train_loader):
    optimizer = torch.optim.Adam(model.parameters())  # (lr=0.001, betas=(0.9,0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    error = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(EPOCHS):
        correct = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            var_X_batch = Variable(X_batch).float()
            var_y_batch = Variable(y_batch)
            optimizer.zero_grad()
            output = model(var_X_batch)
            loss = error(output, var_y_batch)
            loss.backward()
            optimizer.step()

            # Total correct predictions
            predicted = torch.max(output.data, 1)[1]
            correct += (predicted == var_y_batch).sum()
            if batch_idx % 50 == 0:
                print('Epoch : {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy:{:.3f}%'.format(
                    epoch, batch_idx * len(X_batch), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item(), float(correct * 100) / float(BATCH_SIZE * (batch_idx + 1))))


def extract_real_max(output):
    o = output.detach().numpy()
    predicted = []
    for a in o:
        top_three_predictions = a.argsort()[-3:][::-1]
        predicted.append(max(top_three_predictions))
    predicted = torch.from_numpy(np.array(predicted))
    return predicted


def evaluate(model):
    correct = 0
    for test_imgs, test_labels in test_loader:
        test_imgs = Variable(test_imgs).float()
        output = model(test_imgs)

        predicted = extract_real_max(output)
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{:.3f} ".format(float(correct) / (len(test_loader) * BATCH_SIZE)))


torch_X_train = torch_X_train.view(-1, 1, 128, 128).float()
torch_X_test = torch_X_test.view(-1, 1, 128, 128).float()

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)

# data loader
train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Linear(64 * 64 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


cnn = CNN()
fit(cnn, train_loader)
evaluate(cnn)
