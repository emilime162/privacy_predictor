import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import os

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


# def evaluate_model(model, dataset, test_folder_path):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     images_path = os.path.join(test_folder_path, 'images')
#     labels_path = os.path.join(test_folder_path, 'labels')
#     threshold = 0.5
#     num_classes = len(dataset.labels)

#     print(f"Evaluating for threshold: {threshold}")

#     y_true = []
#     y_pred = []

#     for filename in os.listdir(images_path):
#         if filename.endswith(".jpg"):
#             image_path = os.path.join(images_path, filename)
#             label_file = os.path.splitext(filename)[0] + '.json'
#             label_path = os.path.join(labels_path, label_file)

#             if not os.path.exists(label_path):
#                 print(f"Label file for {filename} not found.")
#                 continue

#             predicted_labels = model.predict_labels(image_path, threshold)
#             ground_truth_labels = load_ground_truth_labels(label_path, dataset)

#             y_true.append(ground_truth_labels)
#             y_pred.append(predicted_labels)

#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     # Calculate precision, recall, and F1-score
#     precision = precision_score(y_true, y_pred, average='micro')
#     recall = recall_score(y_true, y_pred, average='micro')
#     f1 = f1_score(y_true, y_pred, average='micro')

#     print(f"Threshold: {threshold:.1f}")
#     print(f"Precision: {precision:.4f}")
#     print(f"Recall: {recall:.4f}")
#     print(f"F1 Score: {f1:.4f}\n")