import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the TimeDistributed class
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # Reshape input to (batch_size * time_steps, input_size)
        batch_size, time_steps, *input_dims = x.size()
        x = x.view(batch_size * time_steps, *input_dims)
        
        # Apply the module
        x = self.module(x)
        
        # Reshape back to (batch_size, time_steps, output_size)
        output_dims = x.size()[1:]  # Shape after applying module
        x = x.view(batch_size, time_steps, *output_dims)
        return x

# Define the model
class YogaPoseClassifier(nn.Module):
    def __init__(self, num_keypoints=33, num_classes=6):
        super(YogaPoseClassifier, self).__init__()
        
        # TimeDistributed 1D CNN layers
        self.time_distributed_1 = TimeDistributed(nn.Conv1d(in_channels=3, out_channels=16, kernel_size=1))
        self.time_distributed_2 = TimeDistributed(nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1))
        self.time_distributed_3 = TimeDistributed(nn.Conv1d(in_channels=16, out_channels=16, kernel_size=1))
        
        # Batch normalization
        self.batch_norm = TimeDistributed(nn.BatchNorm1d(num_features=16))
        
        # Flatten layer before feeding to LSTM
        self.flatten = TimeDistributed(nn.Flatten())
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=num_keypoints * 16, hidden_size=20, batch_first=True)
        
        # Output layer (fully connected with softmax)
        self.time_distributed_5 = TimeDistributed(nn.Linear(20, num_classes))
    
    def forward(self, x):
        # x shape: (batch_size, time_steps, num_keypoints, 3)
        
        # Apply TimeDistributed CNNs
        x = self.time_distributed_1(x)  # (batch_size, time_steps, num_keypoints, 16)
        x = F.relu(x)
        x = self.time_distributed_2(x)  # (batch_size, time_steps, num_keypoints, 16)
        x = F.relu(x)
        x = self.time_distributed_3(x)  # (batch_size, time_steps, num_keypoints, 16)
        
        # Apply batch normalization
        x = self.batch_norm(x)  # (batch_size, time_steps, num_keypoints, 16)
        
        # Flatten the data for LSTM
        x = self.flatten(x)  # (batch_size, time_steps, num_keypoints * 16)
        
        # LSTM layer
        x, (h_n, c_n) = self.lstm(x)  # (batch_size, time_steps, 20)
        
        # Output layer (fully connected with softmax activation)
        x = self.time_distributed_5(x)  # (batch_size, time_steps, num_classes)
        output = F.log_softmax(x, dim=-1)  # Apply softmax for classification
        
        return output

# Model initialization
num_classes = 6  # Number of yoga poses to classify
num_keypoints = 33  # 33 landmarks per frame
model = YogaPoseClassifier(num_keypoints=num_keypoints, num_classes=num_classes)

# Check the model architecture
print(model)
