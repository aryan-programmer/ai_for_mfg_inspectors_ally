import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

from utils.constants import NEG_CLASS
from tqdm.auto import tqdm


# This function takes in the dataloader (for loading training data), the model, optimizer, loss criterion, number of epochs, device to train the model on (CPU or GPU), and an optional target accuracy to stop training early.
def train(
    dataloader, model, optimizer, criterion, epochs, device, target_accuracy=None
):
    """
    Script to train a model. Returns trained model.
    """

    # These lines move the model to the specified device (CPU or GPU) and puts the model in train mode.
    model.to(device)
    model.train()

    # This loop iterates over the number of epochs specified and initializes variables to track loss, accuracy, and number of samples processed during training
    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        running_loss = torch.tensor(0.0).to(device)
        running_corrects = torch.tensor(0.0).to(device)
        n_samples = 0

        # This inner loop iterates over batches of data from the dataloader, moves the inputs and labels to the specified device, performs forward and backward passes through the model, calculates the loss and updates the model weights via the optimizer, and updates the running loss and accuracy statistics.
        for inputs, labels in tqdm(dataloader, desc=f"Epoch {epoch}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            preds_scores = model(inputs)
            preds_class = torch.argmax(preds_scores, dim=-1)
            loss = criterion(preds_scores, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds_class == labels)
            n_samples += inputs.size(0)

        # This code calculates the average loss and accuracy over the epoch and prints the results. If a target accuracy is specified, the code checks if the current epoch's accuracy exceeds the target and stops training early if it does.
        epoch_loss = running_loss / n_samples
        epoch_acc = running_corrects.double() / n_samples
        print(
            f"Epoch {epoch}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {epoch_acc:.4f}"
        )

        if target_accuracy != None:
            if epoch_acc > target_accuracy:
                print("Early Stopping")
                break
        print()

    # This function returns the trained model.
    return model


def plot_confusion_matrix(y_true, y_pred, class_names="auto"):
    # Calculate the confusion matrix using scikit-learn's metrics
    confusion = confusion_matrix(y_true, y_pred)
    # Create a new figure with a specified size
    plt.figure(figsize=[5, 5])
    # Plot the confusion matrix as a heatmap using seaborn's heatmap function
    sns.heatmap(
        confusion,
        annot=True,
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
    )

    # Set the labels and title of the plot
    plt.ylabel("True labels")
    plt.xlabel("Predicted labels")
    plt.title("Confusion Matrix")
    # Display the plot
    plt.show()


def evaluate(model, dataloader, device):
    """
    Script to evaluate a model after training.
    Outputs accuracy and balanced accuracy, draws confusion matrix.
    """
    # Move the model to the specified device
    model.to(device)
    # Set the model to evaluation mode
    model.eval()
    # Get the class names from the dataloader's dataset
    class_names = dataloader.dataset.classes

    # Initialize variables to keep track of correct predictions, true labels, and predicted labels
    y_true = np.empty(shape=(0,))
    y_pred = np.empty(shape=(0,))

    # Loop through the dataloader's batches
    for inputs, labels in tqdm(dataloader):
        # Move the inputs and labels to the specified device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass the inputs through the model and get the predicted probabilities and classes
        preds_probs = model(inputs)[0]
        preds_class = torch.argmax(preds_probs, dim=-1)

        # Move the labels and predicted classes to the CPU and convert them to numpy arrays
        labels = labels.to("cpu").numpy()
        preds_class = preds_class.detach().to("cpu").numpy()

        # Concatenate the true labels and predicted labels to the respective arrays
        y_true = np.concatenate((y_true, labels))
        y_pred = np.concatenate((y_pred, preds_class))

    # Calculate the accuracy and balanced accuracy scores using scikit-learn's metrics
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)

    # Print the accuracy and balanced accuracy scores
    print("Accuracy: {:.4f}".format(accuracy))
    print("Balanced Accuracy: {:.4f}".format(balanced_accuracy))
    print()
    # Plot the confusion matrix using seaborn's heatmap function
    plot_confusion_matrix(y_true, y_pred, class_names=class_names)


def get_bbox_from_heatmap(heatmap, thres=0.8):
    """
    Returns bounding box around the defected area:
    Upper left and lower right corner.

    Threshold affects size of the bounding box.
    The higher the threshold, the wider the bounding box.
    """
    # Create a binary map by thresholding the heatmap
    binary_map = heatmap > thres

    # Compute the x-coordinate of the left and right edge of the bounding box
    x_dim = np.max(binary_map, axis=0) * np.arange(0, binary_map.shape[1])
    x_0 = int(x_dim[x_dim > 0].min())
    x_1 = int(x_dim.max())

    # Compute the y-coordinate of the top and bottom edge of the bounding box
    y_dim = np.max(binary_map, axis=1) * np.arange(0, binary_map.shape[0])
    y_0 = int(y_dim[y_dim > 0].min())
    y_1 = int(y_dim.max())

    # Return the four corners of the bounding box
    return x_0, y_0, x_1, y_1


# The function shows the image, its true label, predicted label and predicted probability.
# If the model predicts an anomaly, the function draws a bounding box (bbox) around the defected region and a heatmap.
# The plot displays the images in a grid, with each image and its label/prediction information in one subplot.
def predict_localize(
    model, dataloader, device, thres=0.8, n_samples=9, show_heatmap=False
):
    """
    Runs predictions for the samples in the dataloader.
    Shows image, its true label, predicted label and probability.
    If an anomaly is predicted, draws bbox around defected region and heatmap.
    """

    # Move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    # Get class names from dataloader
    class_names = dataloader.dataset.classes

    # Convert PyTorch tensor to PIL Image for displaying images
    transform_to_PIL = transforms.ToPILImage()

    # Calculate number of rows and columns for subplot visualization
    n_cols = 5
    n_rows = int(np.ceil(n_samples / n_cols))

    # Set figure size
    plt.figure(figsize=[n_cols * 4, n_rows * 4])

    # Initialize sample counter
    counter = 0

    # Iterate over batches in dataloader
    for inputs, labels in dataloader:

        # Move batch to device
        inputs = inputs.to(device)

        # Generate predictions and feature maps from model
        out = model(inputs)
        probs, class_preds = torch.max(out[0], dim=-1)
        feature_maps = out[1].to("cpu")

        # Iterate over images in batch
        for img_i in range(inputs.size(0)):

            # Get image, predicted label, probability, and true label
            img = transform_to_PIL(inputs[img_i])
            class_pred = class_preds[img_i]
            prob = probs[img_i]
            label = labels[img_i]

            # Get heatmap for negative class (anomaly) if predicted
            heatmap = feature_maps[img_i][NEG_CLASS].detach().numpy()

            # Increment subplot counter
            counter += 1

            # Create subplot for image
            plt.subplot(n_rows, n_cols, counter)

            # Show image and set axis off
            plt.imshow(img)
            plt.axis("off")

            # Set title with predicted label, probability, and true label
            plt.title(
                "\nPredicted: {}, Prob: {:.3f}\nTrue Label: {}".format(
                    class_names[class_pred], prob, class_names[label]
                )
            )

            # If anomaly is predicted (negative class)
            if class_pred == NEG_CLASS:
                # Get bounding box from heatmap and draw rectangle around anomaly
                x_0, y_0, x_1, y_1 = get_bbox_from_heatmap(heatmap, thres)
                rectangle = Rectangle(
                    (x_0, y_0),
                    x_1 - x_0,
                    y_1 - y_0,
                    edgecolor="red",
                    facecolor="none",
                    lw=3,
                )
                plt.gca().add_patch(rectangle)

                # If show_heatmap is True, show heatmap
                if show_heatmap:
                    plt.imshow(heatmap, cmap="Reds", alpha=0.3)

            # If counter equals number of samples, show plot and return
            if counter == n_samples:
                plt.tight_layout()
                plt.show()
                return

    plt.tight_layout()
    plt.show()
