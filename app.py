import io
import os
from collections.abc import Sequence

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torchvision.transforms as transforms
from matplotlib.patches import Rectangle
from PIL import Image
from utils.constants import NEG_CLASS
from utils.dataloader import get_train_test_loaders
from utils.helper import get_bbox_from_heatmap
from utils.model import CustomVGG

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


@st.cache_resource
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_cpu_device() -> torch.device:
    return torch.device("cpu")


DATA_FOLDER = "./data/"
SUBSET_NAME = "tile"
DATA_FOLDER = os.path.join(DATA_FOLDER, SUBSET_NAME)

BATCH_SIZE = 1
HEATMAP_THRESHOLD = 0.5

SUBSET_NAME = "tile"
MODEL_PATH = f"./weights/{SUBSET_NAME}_model.h5"


@st.cache_resource
def get_model() -> CustomVGG:
    model = CustomVGG(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.to(get_device())
    model.eval()
    return model


# Get the list of class names from the test loader
@st.cache_data
def get_class_names() -> Sequence[str]:
    _, test_loader = get_train_test_loaders(DATA_FOLDER, batch_size=BATCH_SIZE)
    return tuple(map(str, test_loader.dataset.classes))


# Define the functions to load images
def load_uploaded_image(file: str) -> Image.Image:
    return Image.open(file)


def buffer_plot_and_get(fig: matplotlib.figure.Figure):
    buf = io.BytesIO()
    fig.savefig(buf, transparent=True)
    buf.seek(0)
    image = Image.open(buf)
    image = image.crop(image.getbbox())
    return image


def anomaly_detection(
    images_and_names: Sequence[tuple[Image.Image, str]],
) -> tuple[Sequence[tuple[Image.Image, str]], Sequence[tuple[Image.Image, str]]]:
    """
    Given an image path and a trained PyTorch model, returns the predicted class and bounding boxes for any defects detected in the image.
    """

    resize = transforms.Resize((224, 224))
    to_tensor = transforms.ToTensor()
    images = torch.tensor(
        np.array([to_tensor(resize(image)) for image, _ in images_and_names])
    ).to(get_cpu_device())
    transform_to_PIL = transforms.ToPILImage()
    model = get_model()
    class_names = get_class_names()

    # Get the model's predictions for the image
    with torch.no_grad():
        images_device = images.to(get_device(), copy=True)
        out = model(images_device)
        feature_maps = out[1].to(get_cpu_device())

        # Get the predicted class label and probability
        probs, pred_class_idxs = torch.max(out[0].to(get_cpu_device()), dim=-1)
        del out

    good_res: list[tuple[Image.Image, str]] = []
    bad_res: list[tuple[Image.Image, str]] = []
    for i, images in enumerate(images):
        predicted_class = class_names[pred_class_idxs[i]]
        # Get heatmap for negative class (anomaly) if predicted
        heatmap = feature_maps[i][NEG_CLASS].detach().numpy()

        fig = plt.figure()

        # Set title with predicted label, probability, and true label
        caption = images_and_names[i][1]

        # If anomaly is predicted (negative class)
        if pred_class_idxs[i] == NEG_CLASS:
            plt.subplot(1, 2, 1)
            plt.title(f"Bounding Box: Probability: {probs[i]:.3f}")

            plt.imshow(transform_to_PIL(images))
            plt.axis("off")
            # Get bounding box from heatmap and draw rectangle around anomaly
            x_0, y_0, x_1, y_1 = get_bbox_from_heatmap(heatmap, HEATMAP_THRESHOLD)
            rectangle = Rectangle(
                (x_0, y_0),
                x_1 - x_0,
                y_1 - y_0,
                edgecolor="red",
                facecolor="none",
                lw=3,
            )
            plt.gca().add_patch(rectangle)

            plt.subplot(1, 2, 2)
            plt.title(f"Original {images_and_names[i][1]} with Heatmap")

            plt.imshow(transform_to_PIL(images))
            plt.axis("off")

            plt.imshow(heatmap, cmap="Reds", alpha=0.3)

            plt.tight_layout()
            bad_res.append((buffer_plot_and_get(fig), caption))
        else:
            plt.imshow(transform_to_PIL(images))
            plt.title(
                f"Good image {images_and_names[i][1]} with Probability {probs[i]:.3f}"
            )
            plt.axis("off")
            plt.tight_layout()
            good_res.append((buffer_plot_and_get(fig), caption))

    return (good_res, bad_res)


# Set up the page layout
st.set_page_config(page_title="InspectorsAlly", page_icon=":camera:")

page_bg_img = """
<style>
[data-testid="stImageContainer"] {
    padding: 4px;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


st.title("InspectorsAlly: Watch Your Step")

st.caption(
    "Boost Your Quality Control with InspectorsAlly - The Ultimate AI-Powered Inspection App"
)

st.write(
    "Try uploading a floor tile image and watch how an AI Model will classify it between Good / Anomaly."
)

with st.sidebar:
    img = Image.open("./docs/overview_dataset.jpg")
    st.image(img)
    st.subheader("About InspectorsAlly: Watch Your Step")
    st.write(
        "InspectorsAlly is a powerful AI-powered application designed to help businesses streamline their quality control inspections. With InspectorsAlly, companies can ensure that their products meet the highest standards of quality, while reducing inspection time and increasing efficiency."
    )

    st.write(
        "This advanced inspection app uses state-of-the-art computer vision algorithms and deep learning models to perform visual quality control inspections with unparalleled accuracy and speed. InspectorsAlly is capable of identifying even the slightest defects, such as scratches, dents, discolorations, and more on the Floor Tile Product Images."
    )

    st.write(
        "This particular version is focused on analysing anomalies in floor tiles."
    )


# Set up the sidebar
st.subheader("Select Image Input Method")
input_method = st.radio(
    "options", ["File Uploader", "Camera Input"], label_visibility="collapsed"
)

camera_file_img = None
uploaded_file_images = ()

# Check which input method was selected
if input_method == "File Uploader":
    uploaded_files = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded_files is not None and len(uploaded_files) > 0:
        uploaded_file_images = tuple(
            (load_uploaded_image(uploaded_file), uploaded_file.name)
            for uploaded_file in uploaded_files
        )
        images, captions = map(list, zip(*uploaded_file_images))
        st.image(images, caption=captions, width=200)
        st.success("Image(s) uploaded successfully!")
    else:
        st.warning("Please upload at least 1 image file.")

elif input_method == "Camera Input":
    st.warning("Please allow access to your camera.")
    camera_image_file = st.camera_input("Click an Image")
    if camera_image_file is not None:
        camera_file_img = (
            load_uploaded_image(camera_image_file),
            "Camera Input",
        )
        st.image(camera_file_img[0], caption="Camera Input Image", width=300)
        st.success("Image clicked successfully!")
    else:
        st.warning("Please click an image.")


submit = st.button(label="Submit Floor Tile Image(s)")
if submit:
    st.subheader("Output")
    img_file_path = ()
    if input_method == "File Uploader" and uploaded_file_images is not None:
        img_file_path = uploaded_file_images
    elif input_method == "Camera Input" and camera_file_img is not None:
        img_file_path = (camera_file_img,)
    if img_file_path is None or len(img_file_path) == 0:
        st.warning("Please specify at least 1 image")
    else:
        with st.spinner(text="This may take a moment..."):
            good_res, bad_res = anomaly_detection(img_file_path)

            def show_good_images(good_res: Sequence[tuple[Image.Image, str]]):
                good_res_even, good_res_odd = good_res[::2], good_res[1::2]

                col1, col2 = st.columns(2)
                with col1:
                    good_images_even, good_captions_even = map(
                        list, zip(*good_res_even)
                    )
                    st.image(good_images_even, caption=good_captions_even)
                if len(good_res_odd) != 0:
                    with col2:
                        good_images_odd, good_captions_odd = map(
                            list, zip(*good_res_odd)
                        )
                        st.image(good_images_odd, caption=good_captions_odd)

            if len(good_res) == 0:
                st.write(
                    "We're sorry to inform you that our AI-based visual inspection system has detected no good products, i.e. all your products are anomalous."
                )
                bad_images, bad_captions = map(list, zip(*bad_res))
                st.image(bad_images, caption=bad_captions, use_container_width=True)
            elif len(bad_res) == 0:
                st.write(
                    "Congratulations! All of your products have been classified as 'Good' items with no anomalies detected in the inspection images."
                )
                show_good_images(good_res)
            else:
                st.write(
                    "Our AI-based visual inspection system has classified all of the below products as 'Good' items with no anomalies detected:"
                )
                show_good_images(good_res)
                st.write(
                    "However, we're sorry to inform you that the below products have an anomaly in thier inspection images."
                )
                bad_images, bad_captions = map(list, zip(*bad_res))
                st.image(bad_images, caption=bad_captions, use_container_width=True)
