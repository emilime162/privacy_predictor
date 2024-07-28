import sys
import os

# Append the current working directory to the system path
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from utils import preprocess_image, load_ground_truth_labels  # Import from utils.py
from data_utils import DataUtils  
import config
from sklearn.metrics import precision_score, recall_score, f1_score

# Check CUDA availability and capability
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 3.7:
    device = torch.device('cuda')
    print("CUDA is available. Device count:", torch.cuda.device_count())
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("CUDA is not available or not supported. Using CPU.")

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

    def evaluate_model(model, dataset, test_folder_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        images_path = os.path.join(test_folder_path, 'images')
        labels_path = os.path.join(test_folder_path, 'labels')
        threshold = 0.5
        num_classes = len(dataset.labels)

        print(f"Evaluating for threshold: {threshold}")

        y_true = []
        y_pred = []

        for filename in os.listdir(images_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(images_path, filename)
                label_file = os.path.splitext(filename)[0] + '.json'
                label_path = os.path.join(labels_path, label_file)

                if not os.path.exists(label_path):
                    print(f"Label file for {filename} not found.")
                    continue

                predicted_labels = model.predict_labels(image_path, threshold)
                ground_truth_labels = load_ground_truth_labels(label_path, dataset)

                y_true.append(ground_truth_labels)
                y_pred.append(predicted_labels)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate precision, recall, and F1-score
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')

        print(f"Threshold: {threshold:.1f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}\n")
    
if __name__ == "__main__":
    # Load the preprocessed dataset
    data_utils = DataUtils('', '')  # Dummy paths, since we're loading a preprocessed dataset
    dataset = data_utils.load_dataset('preprocessed_dataset.pkl')

        # Example of testing the model with data in the test file
    model = MyCNN(num_classes=len(dataset.labels))  # Use the correct number of classes (2 in this case)

    model.load_state_dict(torch.load('multi_label_image_classification_model.pth', map_location=device), strict=False)
    model = model.to(device)

    # Testing the model
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()
    test_loss = model.test_model(test_loader, criterion)
    print(f"Test Loss: {test_loss}")

    # Evaluate the model
    test_folder_path = 'test'
    model.evaluate_model(dataset, test_folder_path)

    # Predicting labels for a single image
    test_image_path = "/home/chen.shix/test/images/2017_10665299.jpg"  # Make sure this path is correct
    threshold = 0.5
    predicted_labels = model.predict_labels(test_image_path, threshold)
    print(f"Predicted labels for {test_image_path}: {predicted_labels}")

    # # Example of testing the model with data in the test file
    # model = MyCNN(num_classes=len(dataset.labels))  # Use the correct number of classes

    # model.load_state_dict(torch.load('multi_label_image_classification_model.pth'), strict=False)
    # model = model.to(config.device)

    # # Testing the model
    # # Create a DataLoader for the test dataset
    # test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    # criterion = nn.BCEWithLogitsLoss()
    # test_loss = model.test_model(test_loader, criterion)
    # print(f"Test Loss: {test_loss}")

    # # Predicting labels for a single image
    # test_image_path = "/home/chen.shix/test/images/2017_10665299.jpg"  
    # threshold = 0.5
    # predicted_labels = model.predict_labels(test_image_path, threshold)
    # print(f"Predicted labels for {test_image_path}: {predicted_labels}")