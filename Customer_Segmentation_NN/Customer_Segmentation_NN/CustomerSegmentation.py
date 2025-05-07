"""
This program develops a simple neural network for predicting what group customers belong
to in an automobile companies customer segmentation.
"""

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def readCustomerData(filePath):
    data = pd.read_csv(filePath)
    data['Segmentation'] = data['Segmentation'].astype('category')
    categories = list(data['Segmentation'].cat.categories)
    data['Segmentation'] = data['Segmentation'].cat.codes
    #need to clean up the features removing profession becuase it is too complex of a category
    data = data.drop('Profession', axis=1)

    #set a one-hot encoding of the remaining categorical variables
    data = pd.get_dummies(data, columns=['Gender', 'Ever_Married', 'Graduated','Spending_Score', 'Var_1'])
    data = data.apply(lambda x: x.astype(float) if x.dtype == 'bool' else x)
    data.dropna(inplace=True)

    X = data.iloc[:, 1:-1].copy().to_numpy()
    y = data['Segmentation'].copy().to_numpy()

    return X, y, categories

class CustomerSegmentation:
    def __init__(self):
        X, y, _ = readCustomerData("Segmentation_Train.csv")

        #creates a tensor ie df of labels and the categories
        self.X = torch.tensor(X, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)

        X, y, _ = readCustomerData("Segmentation_Train.csv")

        self.X_valid = torch.tensor(X, dtype=torch.float)
        self.y_valid = torch.tensor(y, dtype=torch.long)

        self.len = len(self.X)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.len

class SegmentationNN(nn.Module):
    def __init__(self):
        super(SegmentationNN, self).__init__()
        #uses batchNorm1d due to 1 dimensional data
        #19 due to having 19 labels after one hot encoding
        self.norm = nn.BatchNorm1d(19)

        #starting off with one hidden layer, and no dropouts
        self.in_to_h1 = nn.Linear(19, 38)
        self.h1_to_out = nn.Linear(38, 19)

    def forward(self, x):
        # normalized in the nn to tamp down runaway variance in runtime
        x = self.norm(x)
        x = F.relu(self.in_to_h1(x))
        return self.h1_to_out(x)

def trainNN(epochs=10, batch_size=16, lr=0.001, trained_network=None, save_file="segmentationNN.pt"):
    # Create dataset of class CustomerSegmentation
    cs = CustomerSegmentation()

    # Create data loader for use by each epoch
    dl = DataLoader(cs, batch_size=batch_size, shuffle=True, drop_last=True)

    # Create the ANN and load save point from file if it exists
    segmentNN = SegmentationNN()
    if trained_network is not None:
        segmentNN.load_state_dict(torch.load(trained_network))
        segmentNN.train()

    # loss function
    loss_fn = CrossEntropyLoss()

    # Optimizer, selecting Adam optimizer
    optimizer = torch.optim.Adam(segmentNN.parameters(), lr=lr)

    running_loss = 0.0
    for epoch in range(epochs):
        for _, data in enumerate(tqdm(dl)):
            X, y = data
            optimizer.zero_grad()
            output = segmentNN(X)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        with torch.no_grad():
            segmentNN.eval()
            print(f"\nRunning loss for epoch {epoch + 1} of {epochs}: {running_loss:.4f}")
            predictions = torch.argmax(segmentNN(cs.X), dim=1)
            correct = (predictions == cs.y).sum().item()
            print(f"Accuracy on train set: {correct / len(cs.X):.4f}")
            predictions = torch.argmax(segmentNN(cs.X_valid), dim=1)
            correct = (predictions == cs.y_valid).sum().item()
            print(f"Accuracy on validation set: {correct / len(cs.X_valid):.4f}")
            segmentNN.train()
        running_loss = 0.0
    torch.save(segmentNN.state_dict(), save_file)

# Function assesses the ANN on new data it has not seen yet
def testOnNewData(trained_network="segmentationNN.pt"):
    # Get Test Data info for total scoring
    X, y, categories = readCustomerData("Segmentation_Test.csv")

    # Load the segmentation ANN
    segmentNN = SegmentationNN()
    segmentNN.load_state_dict(torch.load(trained_network))
    segmentNN.eval() #

    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    #displays the actual accuracy in the console
    with torch.no_grad():
        predictions = torch.argmax(segmentNN(X), dim=1)
        correct = (predictions == y).sum().item()
        print(f"Accuracy on test set: {correct / len(X):.4f}")

    #displays the confusion matrix of test results
    cm = confusion_matrix(np.array(y), np.array(predictions))
    disp = ConfusionMatrixDisplay(cm, display_labels=categories)
    disp.plot()
    plt.show()

#trainNN(epochs=10, lr = 0.01)
testOnNewData()