import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

data_train = pd.read_csv("Herring/Herring_TRAIN.tsv", sep='\t')
data_test = pd.read_csv("Herring/Herring_TEST.tsv", sep='\t')

X_train = data_train.iloc[:, 1:].values
y_train = data_train.iloc[:, 0].values
X_test = data_test.iloc[:, 1:].values
y_test = data_test.iloc[:, 0].values

class MyFullyConnectedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyFullyConnectedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = 784
hidden_size = 128
num_classes = 2
model = MyFullyConnectedNetwork(input_size, hidden_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        test_accuracy = (predicted == y_test).sum().item() / len(y_test)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.4f}')
