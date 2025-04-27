import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils.constants import (
    GOOD_CLASS_FOLDER,
    DATASET_SETS,
    INPUT_IMG_SIZE,
    IMG_FORMAT,
    NEG_CLASS,
)


class MVTEC_AD_DATASET(Dataset):
    """
    Class to load subsets of MVTEC ANOMALY DETECTION DATASET
    Dataset Link: https://www.mvtec.com/company/research/datasets/mvtec-ad

    Root is path to the subset, for instance, `mvtec_anomaly_detection/tile`
    """

    def __init__(self, root):
        # Define the class labels based on the NEG_CLASS setting.
        self.classes = ["Good", "Anomaly"] if NEG_CLASS == 1 else ["Anomaly", "Good"]
        # Define the image transformation pipeline.
        self.img_transform = transforms.Compose(
            [transforms.Resize(INPUT_IMG_SIZE), transforms.ToTensor()]
        )

        # Load the image filenames and labels for the dataset.
        (
            self.img_filenames,
            self.img_labels,
            self.img_labels_detailed,
        ) = self._get_images_and_labels(root)

    def _get_images_and_labels(self, root):
        # Initialize lists to store image filenames and labels.
        image_names = []
        labels = []
        labels_detailed = []

        # Loop over the dataset sets (e.g., "train", "test") and classes ("good" and "anomaly").
        for folder in DATASET_SETS:
            # Construct the path to the class folder.
            folder = os.path.join(root, folder)

            # Loop over the class folders in the dataset.
            for class_folder in os.listdir(folder):
                # Determine the label for the class based on its folder name.
                label = (
                    1 - NEG_CLASS if class_folder == GOOD_CLASS_FOLDER else NEG_CLASS
                )
                # Store the detailed label (i.e., the class folder name).
                label_detailed = class_folder

                # Construct the path to the class image folder.
                class_folder = os.path.join(folder, class_folder)
                # Get the list of image filenames in the class folder that match the IMG_FORMAT setting.
                class_images = os.listdir(class_folder)
                class_images = [
                    os.path.join(class_folder, image)
                    for image in class_images
                    if image.find(IMG_FORMAT) > -1
                ]

                # Add the class image filenames and labels to the respective lists.
                image_names.extend(class_images)
                labels.extend([label] * len(class_images))
                labels_detailed.extend([label_detailed] * len(class_images))

        # Print some statistics about the dataset.
        print(
            "Dataset {}: N Images = {}, Share of anomalies = {:.3f}".format(
                root, len(labels), np.sum(labels) / len(labels)
            )
        )
        # Return the lists of image filenames and labels.
        return image_names, labels, labels_detailed

    def __len__(self):
        # Return the length of the dataset (i.e., the number of images).
        return len(self.img_labels)

    def __getitem__(self, idx):
        # Get the filename and label for the image at the specified index.
        img_fn = self.img_filenames[idx]
        label = self.img_labels[idx]
        # Open the image file and apply the image transformation pipeline.
        img = Image.open(img_fn)
        img = img.convert("RGB")
        img = self.img_transform(img)
        # Convert the label to a PyTorch tensor.
        label = torch.as_tensor(label, dtype=torch.long)
        # Return the transformed image and label as a tuple.
        return img, label


# This function takes in the root directory of the MVTEC_AD dataset, batch size for DataLoader, test_size, and random_state as input arguments.
def get_train_test_loaders(root, batch_size, test_size=0.2, random_state=42):
    """
    Returns train and test dataloaders.
    Splits dataset in stratified manner, considering various defect types.
    """
    # Initialize the dataset object with the given root directory.
    dataset = MVTEC_AD_DATASET(root=root)

    # Split the indices of dataset into train and test sets in a stratified manner based on the defect types.
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)),
        test_size=test_size,
        shuffle=True,
        stratify=dataset.img_labels_detailed,
        random_state=random_state,
    )

    # Initialize the SubsetRandomSampler for the training set and test set.
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # Initialize the DataLoader objects for the training set and test set with the SubsetRandomSampler.
    train_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True
    )
    test_loader = DataLoader(
        dataset, batch_size=batch_size, sampler=test_sampler, drop_last=False
    )

    # Return the DataLoader objects for the training set and test set.
    return train_loader, test_loader
