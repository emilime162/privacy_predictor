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
from torch.utils.data import DataLoader  # Ensure DataLoader is imported
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

    def train_model(self, train_loader, val_loader, criterion, optimizer, num_epochs):
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            val_loss = self.validate_model(val_loader, criterion)
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {running_loss/len(train_loader):.4f}, Validation Loss: {val_loss:.4f}")

    def validate_model(self, val_loader, criterion):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        self.model.train()
        return val_loss / len(val_loader)

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
    # Define paths to the preprocessed dataset files
    train_dataset_path = 'train_preprocessed_dataset.pkl'
    val_dataset_path = 'val_preprocessed_dataset.pkl'
    test_folder_path = 'test'  # Adjust this path if necessary

    # Initialize DataUtils and load the preprocessed datasets
    data_utils = DataUtils('', '')  # Dummy paths for initialization
    train_dataset = data_utils.load_dataset(train_dataset_path)
    val_dataset = data_utils.load_dataset(val_dataset_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model, criterion, and optimizer
    model = MyCNN(num_classes=len(train_dataset.labels))
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 25
    model.train_model(train_loader, val_loader, criterion, optimizer, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), 'mobilenet_v2_model.pth')

    # Evaluate the model on the validation set
    val_loss = model.test_model(val_loader, criterion)
    print(f"Validation Loss: {val_loss}")
    
    # for a separate test set, load it similarly and create a DataLoader for it
    # test_dataset = data_utils.load_dataset('test_preprocessed_dataset.pkl')  
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    # test_loss = model.test_model(test_loader, criterion)
    # print(f"Test Loss: {test_loss}")

    # Evaluate the model on the test set using the `evaluate_model` method
    model.evaluate_model(val_dataset, test_folder_path)  # Use appropriate dataset if you have a separate test dataset

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