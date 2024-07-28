import torch

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
k_fold = 10
learning_rate = 0.001
optimizer = "Adam"
