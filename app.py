from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import torch
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from torchvision import transforms
from unet import UNet  # assuming your U-Net model is in unet.py

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join("static", "uploads")
SEGMENTED_FOLDER = r"C:\Users\Abhinav Singh\Desktop\lung_segmentation_project\segmented_output\new segment"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet_lung_segmentation.pth"
model = UNet(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Transformation for the image
IMAGE_SIZE = (256, 256)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def segment_lung(image_path, output_path):
    # Load the image
    image = Image.open(image_path).convert("L").resize(IMAGE_SIZE)
    image_np = np.array(image).astype(np.uint8)

    # Apply the transformation
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # Perform segmentation
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    # Apply the predicted mask to the original image
    # Convert the mask to 0 and 1 values (binary mask)
    mask = np.clip(pred_mask, 0, 1)

    # Create a highlighted version of the original image with the mask applied
    highlighted_image = image_np * mask

    # Increase the brightness of the resulting image
    # Increase brightness by scaling pixel values (e.g., 1.2 to increase brightness by 20%)
    brightness_factor = 1.5
    highlighted_image = np.clip(highlighted_image * brightness_factor, 0, 255)  # Ensure pixel values are within valid range

    # Ensure the resulting image is in the correct format (uint8, mode 'L' for grayscale)
    highlighted_image = np.array(highlighted_image, dtype=np.uint8)

    # Save the segmented image (with the mask applied) as PNG
    highlighted_image_pil = Image.fromarray(highlighted_image)
    highlighted_image_pil = highlighted_image_pil.convert("L")  # Ensure it's in grayscale mode 'L'
    highlighted_image_pil.save(output_path)  # Save the image in PNG format

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return "No file part"
        file = request.files["image"]
        if file.filename == "":
            return "No selected file"
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            uploaded_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(uploaded_path)

            # Construct the segmented image filename
            segmented_filename = f"segmented_lung_{filename}"
            segmented_path = os.path.join(SEGMENTED_FOLDER, segmented_filename)

            # Run segmentation automatically
            segment_lung(uploaded_path, segmented_path)

            # Check if the segmented image was created
            if os.path.exists(segmented_path):
                return render_template("index.html", filename=filename, segmented_filename=segmented_filename)
            else:
                return render_template("index.html", filename=filename, segmented_filename=None)

    return render_template("index.html")

@app.route("/segmented/<filename>")
def segmented(filename):
    return send_from_directory(SEGMENTED_FOLDER, filename)

@app.route("/uploaded/<filename>")
def static_uploaded(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
