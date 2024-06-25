import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau

class CustomDataset(Dataset):
    # Initialize an instance of the CustomDataset class
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe  # Store the input dataframe
        self.transform = transform  # Store any transformation operations if provided

    # Return the length of the dataset
    def __len__(self):
        return len(self.dataframe)

    # Get a sample from the dataset based on index idx
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]  # Get the data of the ith row
        target = '紫外线（指数）'  # The name of the target column

        # Convert the features excluding the target column into a tensor
        features = torch.tensor(row.drop(target).values, dtype=torch.float32)
        # Convert the value of the target column into a tensor
        target = torch.tensor(row[target], dtype=torch.float32)

        # If a transformation is defined, apply it to the features
        if self.transform:
            features = self.transform(features)
        
        # Return the features and the target
        return features, target

class TransformerForRegression(nn.Module):
    # Initialize the Transformer model for regression
    def __init__(self, input_dim, num_heads, num_layers, d_model, dim_feedforward, dropout=0.1):
        super(TransformerForRegression, self).__init__()
        
        # Project input to the d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Define the Transformer encoder layer and the number of layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Define two hidden layers with ReLU activation and batch normalization
        self.layer1 = nn.Linear(d_model, dim_hidden1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(dim_hidden1)

        self.layer2 = nn.Linear(dim_hidden1, dim_hidden2)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(dim_hidden2)
        
        # Final linear layer for regression output
        self.fc = nn.Linear(dim_hidden2, 1)

    # Forward pass through the network
    def forward(self, x):
        # Project input to d_model dimension and pass through Transformer encoder
        x = self.input_projection(x)  
        x = self.transformer_encoder(x)  

        # First hidden layer with batch normalization and ReLU activation
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Second hidden layer with batch normalization and ReLU activation
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        # Final regression output layer
        out = self.fc(out)
        return out

# Define the dimensions and hyperparameters for the model
input_dim = 13  # Dimension of input features
d_model = 128   # Dimension of features in Transformer
num_heads = 8   # Number of attention heads in Transformer
num_layers = 4  # Number of layers in Transformer encoder
dim_feedforward = 256  # Dimension of the feedforward network in Transformer

# Dimensions of the neural network hidden layers
dim_hidden1 = 256  
dim_hidden2 = 512

# Load the preprocessed data into a pandas DataFrame
df = pd.read_csv('preprocess1.csv')

# Define the target variable
target = '紫外线（指数）'

# Separate the target variable and the features
labels = df[target]
data = df.drop(columns=[target])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create DataFrames for training and testing data with the target variable
train_combined = pd.DataFrame(X_train, columns=data.columns)
train_combined[target] = y_train.reset_index(drop=True)

test_combined = pd.DataFrame(X_test, columns=data.columns)
test_combined[target] = y_test.reset_index(drop=True)

# Create dataset objects for training and testing
train_dataset = CustomDataset(train_combined)
test_dataset = CustomDataset(test_combined)

# Create an instance of the Transformer model for regression
model = TransformerForRegression(input_dim, num_heads, num_layers, d_model, dim_feedforward)

# Define the loss function (Mean Squared Error Loss for regression)
criterion = nn.MSELoss()

# Define the optimizer (Adam) with learning rate and weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# Define the learning rate scheduler (ReduceLROnPlateau) to adjust the learning rate during training
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1)

# Create DataLoaders for training and testing datasets
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define the number of epochs for training
num_epochs = 20

# Set the model to training mode
model.train()

# Training loop
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # Zero the gradients before backward pass
        optimizer.zero_grad()

        # Forward pass: pass inputs through the model to get outputs
        outputs = model(inputs).flatten()
        
        # Calculate the loss between predicted outputs and actual labels
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        
        # Update the model parameters based on the gradients
        optimizer.step()
    
        # Print the training progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
    
    # Step the scheduler to adjust the learning rate based on the loss
    scheduler.step(loss)

# Switch the model to evaluation mode
model.eval()

# Use torch.no_grad() to disable gradient computation, saving memory and computations
with torch.no_grad():
    # Initialize lists to store scores for evaluation metrics
    r2_scores = []
    mse_scores = []
    mae_scores = []
    rmse_scores = []

    # Iterate over the test data loader
    for inputs, labels in test_loader:
        # Get model predictions for the batch
        outputs = model(inputs)

        # Convert labels and outputs to numpy arrays for metric calculation
        labels = labels.numpy()
        outputs = outputs.view(-1).numpy()

        # Calculate and store the evaluation metrics for the batch
        r2_scores.append(r2_score(labels, outputs))
        mse_scores.append(mean_squared_error(labels, outputs))
        mae_scores.append(mean_absolute_error(labels, outputs))
        rmse_scores.append(np.sqrt(mean_squared_error(labels, outputs)))
        
    # Calculate the mean of the evaluation metrics across all batches
    r2 = np.mean(r2_scores)
    msee = np.mean(mse_scores)
    mae = np.mean(mae_scores)
    rmse = np.mean(rmse_scores)

    # Print the evaluation metrics
    print(f'MSE Score: {msee:.4f}')
    print(f'MAE Score: {mae:.4f}')
    print(f'RMSE Score: {rmse:.4f}')
    print(f'R^2 Score: {r2:.4f}')