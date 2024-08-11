import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import os
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def load_ground_truth_labels(json_path, dataset):
    with open(json_path, 'r') as file:
        data = json.load(file)
    labels = data['labels']
    ground_truth_labels = [0] * len(dataset.labels)
    for label in labels:
        if label in dataset.label_to_idx:
            ground_truth_labels[dataset.label_to_idx[label]] = 1
    return ground_truth_labels



def plot_and_save_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png')  # Save the plot as an image file
    plt.close()  # Close the plot to avoid displaying it in the terminal
