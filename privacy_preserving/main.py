import os
import torch
from data_utils import DataUtils, save_dataframe_to_csv
from model import MyCNN
import config
from utils import evaluate_model, preprocess_image

# Define paths to the folders
label_folder_path = 'training/labels'
image_folder_path = 'training/images'

# Initialize DataUtils and load data
data_utils = DataUtils(label_folder_path, image_folder_path)
df = data_utils.load_json_files_to_df()

# Optionally save the DataFrame to a CSV file for further inspection
save_dataframe_to_csv(df, 'image_data_and_labels.csv')

# Create the dataset
transform = data_utils.get_transform()
dataset = DataUtils.MultiLabelImageDataset(df, transform=transform)

# # Print dataset information
# print(f"Number of samples in dataset: {len(dataset)}")
# print(f"Example data point: {dataset[0]}")

# Visualize an example image
data_utils.visualize_image(dataset, 0)

# Instantiate the model
model = MyCNN(num_classes=len(dataset.labels))
model.load_state_dict(torch.load('multi_label_image_classification_model.pth'))
model = model.to(config.device)

# Train and evaluate the model
if __name__ == "__main__":

    # Evaluate the model on test data
    test_folder_path = 'test'
    evaluate_model(model, dataset, test_folder_path)
