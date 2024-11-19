import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Normalize the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

df = pd.read_pickle("Test5/Data/train_dax_data.pkl")
df["Y_scaled"] = scaler_y.fit_transform(df["Y"].values.reshape(-1, 1))
print(df.dtypes)
X = scaler_X.fit_transform(df.iloc[:, 2:-2])  # Exclude 'Y' and unnecessary columns

# Create sequences
def create_sequences(data, labels, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length])
    return np.array(X), np.array(y)

seq_size = 30
X_sequences, y_sequences = create_sequences(X, df["Y_scaled"].values, seq_size)

# Dataset and DataLoader
class FinanceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

dataset = FinanceDataset(X_sequences, y_sequences)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define the LSTM model
class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout=0.2):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Output from the last timestep
        return x

# Initialize model, loss function, and optimizer
input_size = X.shape[1]
output_size = 1
hidden_size = 100
num_layers = 2

model = Net(input_size, output_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(-1), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Plot training loss
plt.plot(losses, label='Training Loss')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), "../Models/best_model.pt")

# Evaluate the model
model.eval()
test_df = pd.read_pickle("../Data/test_dax_data.pkl")
test_X = scaler_X.transform(test_df.iloc[:, :-2])  # Normalize test data
test_y = scaler_y.transform(test_df["Y"].values.reshape(-1, 1))

test_sequences, test_labels = create_sequences(test_X, test_y.flatten(), seq_size)
test_dataset = FinanceDataset(test_sequences, test_labels)
test_loader = DataLoader(test_dataset, batch_size=16)

predictions, actuals = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predictions.append(outputs.numpy())
        actuals.append(labels.numpy())

predictions = scaler_y.inverse_transform(np.concatenate(predictions))
actuals = scaler_y.inverse_transform(np.concatenate(actuals))

# Plot predictions vs actuals
plt.plot(actuals, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()
