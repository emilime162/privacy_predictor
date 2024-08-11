import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import plot_and_save_losses

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
from torchmetrics.classification import MultilabelAccuracy



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
        self.num_classes = num_classes

        self.model = models.mobilenet_v2(pretrained=True)

        # Freeze early layers
        for param in self.model.features.parameters():
            param.requires_grad = False
        # Modify the classifier
        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.model.last_channel, num_classes)
        )


        
    def forward(self, x):
        return self.model(x)


    def train_model(self, train_loader, val_loader, criterion, optimizer, num_epochs, patience=4):
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False

        # Lists to store losses for plotting
        training_losses = []
        validation_losses = []

        # Initialize accuracy metric for training
        train_accuracy_metric = MultilabelAccuracy(num_labels=self.num_classes).to(device)

        for epoch in range(num_epochs):
            if early_stop:
                print(f"Early stopping triggered. Stopping training at epoch {epoch+1}.")
                break
            
            self.model.train()
            running_loss = 0.0
            train_accuracy_metric.reset()

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                # Update accuracy metric
                train_accuracy_metric.update(torch.sigmoid(outputs), labels)

            train_loss = running_loss / len(train_loader)
            train_accuracy = train_accuracy_metric.compute().item()  # Calculate training accuracy

            val_loss, val_accuracy = self.validate_model(val_loader, criterion,self.num_classes)

            # Store the losses
            training_losses.append(train_loss)
            validation_losses.append(val_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy*100:.2f}%, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save(self.state_dict(), 'best_model.pth')  # Save the best model
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"No improvement for {patience} consecutive epochs. Stopping early.")
                    early_stop = True

        # After training, plot and save the losses
        plot_and_save_losses(training_losses, validation_losses)




    def validate_model(self, val_loader, criterion, num_classes):
        self.model.eval()
        val_loss = 0.0
        accuracy_metric = MultilabelAccuracy(num_labels=num_classes).to(device)

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy for multilabel classification
                predictions = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
                accuracy_metric.update(predictions, labels)

        val_loss /= len(val_loader)
        accuracy = accuracy_metric.compute()

        self.model.train()
        return val_loss, accuracy

    # def test_model(self, test_loader, criterion):
    #     self.model.eval()
    #     test_loss = 0.0
    #     with torch.no_grad():
    #         for images, labels in test_loader:
    #             images, labels = images.to(config.device), labels.to(config.device)
    #             outputs = self.forward(images)
    #             loss = criterion(outputs, labels)
    #             test_loss += loss.item()
    #     return test_loss / len(test_loader)



    # Function to debug predictions and ground truth
    def debug_predictions(self, dataloader, num_images=5):
        model.eval()
        with torch.no_grad():
            # Iterate through the dataset
            for i, (images, labels) in enumerate(dataloader):
                if i >= num_images:
                    break
                
                # Move images to the appropriate device
                images = images.to(device)
                
                # Get model predictions
                outputs = model(images)
                predictions = torch.sigmoid(outputs) > 0.5  # Apply sigmoid and threshold
                
                # Move predictions and labels back to CPU for easy handling
                predictions = predictions.cpu().numpy()
                labels = labels.cpu().numpy()
                
                print(f"Image {i+1}:")
                print(f"Predicted Labels: {predictions[0]}")  # Print the predicted labels for the first image in the batch
                print(f"Ground Truth Labels: {labels[0]}")   # Print the ground truth labels for the first image in the batch
                print("-----")

    def evaluate_model(self, test_loader, criterion, num_classes, class_names):
        self.model.eval()
        test_loss = 0.0
        accuracy_metric = MultilabelAccuracy(num_labels=num_classes).to(device)

        y_true = []
        y_pred = []

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.forward(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                # Convert logits to binary predictions
                predictions = torch.sigmoid(outputs) > 0.5  
                accuracy_metric.update(predictions, labels)

                # Store the predictions and true labels for additional metrics
                y_true.append(labels.cpu().numpy())
                y_pred.append(predictions.cpu().numpy())

        test_loss /= len(test_loader)
        accuracy = accuracy_metric.compute().item()

        # Concatenate all batches into a single array
        y_true = np.concatenate(y_true, axis=0)
        y_pred = np.concatenate(y_pred, axis=0)

        # Calculate additional metrics
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')
        hamming = hamming_loss(y_true, y_pred)
        hamming_accuracy = 1 - hamming

        print(f"Hamming Accuracy: {hamming_accuracy:.4f}")
        print(f"Hamming Loss: {hamming:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}\n")
        
        # Generate a confusion matrix for each label
        for i, label in enumerate(class_names):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f'Not {label}', label])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f'Confusion Matrix for {label}')
            plt.savefig(f'confusion_matrix_{label}.png')  # Save the plot as an image file
            plt.close()  # Close the plot to avoid displaying it in the terminal

        return test_loss, accuracy, precision, recall, f1, hamming_accuracy
    
if __name__ == "__main__":
    # # Define paths to the preprocessed dataset files
    train_dataset_path = 'train_preprocessed_dataset.pkl'
    val_dataset_path = 'val_preprocessed_dataset.pkl'

    # # Initialize DataUtils and load the preprocessed datasets
    data_utils = DataUtils('', '')  # Dummy paths for initialization
    train_dataset = data_utils.load_dataset(train_dataset_path)
    val_dataset = data_utils.load_dataset(val_dataset_path)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Count the number of images for each label
    label_columns = ['a16_race', 'a6_hair_color']   # Replace with your actual label columns
    label_counts = {label: 0 for label in label_columns}
    
    # Loop through the entire dataset to count labels
    for _, labels in val_loader:
        for i, label in enumerate(label_columns):
            label_counts[label] += labels[:, i].sum().item()  # Sum the occurrences of each label in the batch

    print("Number of images for each label:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

    # Assuming a batch of data
    for images, labels in val_loader:
        print("label the first set of label")
        print(labels[0])  # Print the first set of labels in the batch
        break  # Only need to check the first batch

    # Initialize model, criterion, and optimizer
    model = MyCNN(num_classes=len(train_dataset.labels))
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 8
    model.train_model(train_loader, val_loader, criterion, optimizer, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), 'mobilenet_v2_model.pth')

    num_classes = 2

    # Load the model
    model = MyCNN(num_classes)  # Make sure this matches the structure when you saved it
    model.load_state_dict(torch.load('mobilenet_v2_model.pth'))

    # If using GPU, move to GPU
    model = model.to('cuda') if torch.cuda.is_available() else model

    # Debug predictions for 5 specific images
    model.debug_predictions( val_loader, num_images=5)


    # Set model to evaluation mode
    model.eval()

    
    # for a separate test set, load it similarly and create a DataLoader for it
    # test_dataset = data_utils.load_dataset('test_preprocessed_dataset.pkl')  
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    # test_loss = model.test_model(test_loader, criterion)
    # print(f"Test Loss: {test_loss}")
    
    # val_loss, val_accuracy = model.validate_model(val_loader, criterion,model.num_classes)
    # print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy*100:.2f}%")


    class_names = ["Race", "hair color"]
    test_loss, test_accuracy, precision, recall, f1, hamming_accuracy = model.evaluate_model(val_loader, criterion, num_classes, class_names)

    # # Predicting labels for a single image
    # test_image_path = "/home/chen.shix/test/images/2017_10665299.jpg"  # Make sure this path is correct
    # threshold = 0.5
    # predicted_labels = model.predict_labels(test_image_path, threshold)
    # print(f"Predicted labels for {test_image_path}: {predicted_labels}")

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