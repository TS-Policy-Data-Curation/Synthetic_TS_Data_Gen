import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# ------------------
# 1) LOAD THE DATA
# ------------------

# Paths to your .npz files
# - Real data (macroeconomic data)
# - Generated data (from a generative model, e.g., GRU or another approach)
real_data_path = "data/macroeconomic.npz"
gen_data_path = "data/macroeconomic_generated_data.npz"

# Load .npz files
# Each file is expected to contain an array accessible under the key 'data'
real_data = np.load(real_data_path)['data']
gen_data = np.load(gen_data_path)['data']

print("Real data shape:", real_data.shape)
print("Generated data shape:", gen_data.shape)

# --------------------------------------
# 2) LABEL AND COMBINE REAL & GENERATED
# --------------------------------------
# Label real data as 0, generated data as 1
real_labels = np.zeros((real_data.shape[0],), dtype=int)
gen_labels = np.ones((gen_data.shape[0],), dtype=int)

# Concatenate along the sample dimension
X = np.concatenate([real_data, gen_data], axis=0)
y = np.concatenate([real_labels, gen_labels], axis=0)

print("Combined X shape:", X.shape)
print("Combined y shape:", y.shape)

# At this point, X should have shape: (total_samples, timesteps, features)
# For example, if real_data is (64, 12, 3) and gen_data is (36, 12, 3),
# combined X is (100, 12, 3).

# ----------------------
# 3) TRAIN/TEST SPLIT
# ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert Numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create Datasets and DataLoaders
batch_size = 64

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------
# 4) DEFINE THE GRU MODEL
# -------------------------
class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUClassifier, self).__init__()

        # GRU layer
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          batch_first=True)

        # Dropout after GRU
        self.dropout1 = nn.Dropout(p=0.4)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, input_size)
        """
        # GRU output: (batch_size, sequence_length, hidden_size)
        gru_out, hidden = self.gru(x)

        # We only need the last time step from GRU for classification
        last_time_step = gru_out[:, -1, :]  # shape: (batch_size, hidden_size)

        x = self.dropout1(last_time_step)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        # For binary classification, we use a single output logit
        return x

# -------------------
# 5) TRAIN THE MODEL
# -------------------
gru_units = 128
model = GRUClassifier(input_size=X.shape[2], hidden_size=gru_units)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 1000
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_x, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_x)  # shape: (batch_size, 1)

        # Convert batch_y to float for BCEWithLogitsLoss
        loss = criterion(outputs.squeeze(), batch_y.float())

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        # Compute accuracy
        predicted = (torch.sigmoid(outputs) >= 0.5).long().squeeze()
        correct += (predicted == batch_y).sum().item()
        total_samples += batch_y.size(0)

    epoch_loss = total_loss / total_samples
    epoch_acc = correct / total_samples

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# ----------------------
# 6) EVALUATE THE MODEL
# ----------------------
model.eval()
test_loss = 0.0
correct = 0
total_samples = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(), batch_y.float())
        test_loss += loss.item() * batch_x.size(0)

        predicted = (torch.sigmoid(outputs) >= 0.5).long().squeeze()
        correct += (predicted == batch_y).sum().item()
        total_samples += batch_y.size(0)

test_loss /= total_samples
test_accuracy = correct / total_samples
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

discriminator_score = test_accuracy - 0.5
print(f"Discriminator Score: {discriminator_score:.4f}")