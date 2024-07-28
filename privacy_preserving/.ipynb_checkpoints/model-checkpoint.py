import sys
import os

# Append the current working directory to the system path
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import torch
import torch.nn as nn
from torchvision import models
from utils import preprocess_image
import config


class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.classifier[1] = nn.Linear(self.model.last_channel, num_classes)

    def forward(self, x):
        return self.model(x)

    def train_model(self, train_loader, criterion, optimizer):
        self.model.train()
        for images, labels in train_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = self.forward(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    def test_model(self, test_loader, criterion):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(config.device), labels.to(config.device)
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        return test_loss / len(test_loader)

    def predict_labels(self, image_path, threshold):
        image = preprocess_image(image_path)
        image = image.to(config.device)
        self.eval()
        with torch.no_grad():
            output = self(image)
        output = torch.sigmoid(output)
        output = output.cpu().numpy()
        predicted_labels = [1 if output[0][i] >= threshold else 0 for i in range(len(output[0]))]
        return predicted_labels

if __name__ == "__main__":
    # Example of testing the model with fake data
    print("start")
    model = MyCNN(num_classes=10)  # Example number of classes
    example_image = torch.rand(1, 3, 224, 224)  # Example fake image
    model.eval()
    with torch.no_grad():
        output = model(example_image)
    print(output)
