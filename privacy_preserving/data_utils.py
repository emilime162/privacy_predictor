import os
import json
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle



# To handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

class DataUtils:
    def __init__(self, label_folder_path, image_folder_path):
        self.label_folder_path = label_folder_path
        self.image_folder_path = image_folder_path
        self.target_labels = ['a16_race', 'a6_hair_color']  # Specify the target labels

    def load_json_files_to_df(self):
        data_list = []

        for filename in os.listdir(self.label_folder_path):
            if filename.endswith(".json"):
                file_path = os.path.join(self.label_folder_path, filename)
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                    image_path = os.path.join(self.image_folder_path, os.path.basename(json_data["image_path"]))
                    filtered_labels = [label for label in json_data["labels"] if label in self.target_labels]
                    if filtered_labels:
                        for label in filtered_labels:
                            data_list.append({
                                "id": json_data["id"],
                                "image_path": image_path,
                                "label": label
                            })

        df = pd.DataFrame(data_list)
        return df

    class MultiLabelImageDataset(Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe.drop_duplicates(subset=['image_path']).reset_index(drop=True)  # Ensure unique images
            self.transform = transform
            self.labels = list(set(dataframe['label']))
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}

            # Create a mapping from image path to a list of labels
            self.image_to_labels = dataframe.groupby('image_path')['label'].apply(list).to_dict()

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            image_path = self.dataframe.iloc[idx]['image_path']
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            labels = self.image_to_labels[image_path]
            label_indices = [self.label_to_idx[label] for label in labels]
            target = np.zeros(len(self.labels))
            target[label_indices] = 1
            return image, torch.tensor(target, dtype=torch.float32)

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def visualize_image(self, dataset, idx):
        image, labels = dataset[idx]
        image = image.numpy().transpose((1, 2, 0))  # Convert from (C, H, W) to (H, W, C)

        # Undo the normalization for visualization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        # Get the label names
        label_names = [dataset.labels[i] for i in range(len(labels)) if labels[i] == 1]

        # Plot the image
        plt.imshow(image)
        plt.title(f'Labels: {", ".join(label_names)}')
        plt.axis('off')
        plt.show()

    def save_dataset(self, dataset, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

    def load_dataset(self, filename):
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
        return dataset

    # Optionally save the DataFrame to a CSV file for further inspection
    def save_dataframe_to_csv(self, df, filename):
        df.to_csv(filename, index=False)

# Test the DataUtils class
if __name__ == "__main__":
    # Define paths to the folders
    train_label_folder_path = 'training/labels'
    train_image_folder_path = 'training/images'
    val_label_folder_path = 'validation/labels'
    val_image_folder_path = 'validation/images'

    # Initialize DataUtils and load data
    train_data_utils = DataUtils(train_label_folder_path, train_image_folder_path)
    val_data_utils = DataUtils(val_label_folder_path, val_image_folder_path)

    # Load data from JSON files into DataFrames
    df_train = train_data_utils.load_json_files_to_df()
    df_val = val_data_utils.load_json_files_to_df()

    # Show the first few rows of the DataFrame to verify data loading
    print("First few rows of the training DataFrame:")
    print(df_train.head())
    print("First few rows of the validation DataFrame:")
    print(df_val.head())

    # Print the number of samples loaded
    print(f"Number of training samples loaded: {len(df_train)}")
    print(f"Number of validation samples loaded: {len(df_val)}")

    # Optionally save the DataFrame to a CSV file for further inspection
    train_data_utils.save_dataframe_to_csv(df_train, 'filtered_image_data_and_labels_train.csv')
    val_data_utils.save_dataframe_to_csv(df_val, 'filtered_image_data_and_labels_val.csv')

    # Create the datasets using the loaded DataFrames
    transform = train_data_utils.get_transform()
    train_dataset = DataUtils.MultiLabelImageDataset(df_train, transform=transform)
    val_dataset = DataUtils.MultiLabelImageDataset(df_val, transform=transform)
 

    # Save the preprocessed dataset
    train_data_utils.save_dataset(train_dataset, 'train_preprocessed_dataset.pkl')
    val_data_utils.save_dataset(val_dataset, 'val_preprocessed_dataset.pkl')


    # Visualize an example image
    train_data_utils.visualize_image(train_dataset, 0)