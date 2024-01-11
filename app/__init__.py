from flask import Flask, render_template, request, url_for
import numpy as np
from ultralytics import YOLO
import os
import random
import string
import cv2

app = Flask(__name__)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


@app.route("/")
def index():
    images_directory = "app/static/images"
    image_filenames = [
        filename
        for filename in os.listdir(images_directory)
        if filename.endswith((".jpg", ".jpeg", ".png", ".gif"))
    ]

    # Generate URLs for the images using url_for
    image_urls = [
        url_for("static", filename=f"images/{filename}") for filename in image_filenames
    ]

    return render_template("index.html", image_urls=image_urls)


@app.route("/detect", methods=["POST"])
def detect():
    weights_path = "app/static/best.pt"
    model = YOLO(weights_path)

    image = request.files["image"].read()

    image_filename = generate_random_filename()
    image_path = os.path.join("app/static/images", image_filename)

    os.makedirs("app/static/images", exist_ok=True)

    output_directory = "app/static/images/output/predict/"
    os.makedirs(output_directory, exist_ok=True)

    render_directory = "images/output/predict"
    with open(image_path, "wb") as temp_image:
        temp_image.write(image)

    results = model.predict(image_path)

    draw_and_save_bounding_boxes(image_path, results, output_directory)

    max_index = np.argmax(results[0].boxes.conf)

    image = cv2.imread(image_path)
    x, y, w, h = results[0].boxes.xywh[max_index]

    x, y, w, h = int(x.item()), int(y.item()), int(w.item()), int(h.item())

    # Calculate the coordinates for cropping
    left = max(0, x - w // 2)
    top = max(0, y - h // 2)
    right = min(image.shape[1], x + w // 2)
    bottom = min(image.shape[0], y + h // 2)

    # Crop the image
    cropped_image = image[top:bottom, left:right]

    # Save the cropped image
    cv2.imwrite(output_directory + image_filename + "_cropped_image.jpg", cropped_image)

    return render_template(
        "result.html", image_path=render_directory + "/" + image_filename
    )


def generate_random_filename():
    random_chars = "".join(random.choices(string.ascii_letters + string.digits, k=8))
    filename = f"{random_chars}.jpg"
    return filename


def draw_and_save_bounding_boxes(image_path, results, output_directory="output"):
    os.makedirs(output_directory, exist_ok=True)
    image = cv2.imread(image_path)
    num_classes = len(set(results[0].boxes.cls))
    colors = list(mcolors.TABLEAU_COLORS.values())[:num_classes]
    for box, label, conf in zip(
        results[0].boxes.xyxy.cpu().numpy(),
        results[0].boxes.cls.cpu().numpy(),
        results[0].boxes.conf.cpu().numpy(),
    ):
        x, y, x_max, y_max = map(int, box)
        (label, "printing label")
        label = int(label)
        color = mcolors.hex2color(colors[label])
        color = tuple(int(c * 255) for c in color)  # Convert to BGR format
        thickness = 8
        font_size = 5
        text = f"{results[0].names[label]}: {conf:.2f}"  # Display class name and confidence
        image = cv2.rectangle(image, (x, y), (x_max, y_max), color, thickness)
        image = cv2.putText(
            image,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_size,
            color,
            thickness,
            cv2.LINE_AA,
        )
    output_path = os.path.join(output_directory, os.path.basename(image_path))
    cv2.imwrite(output_path, image)

    print(f"Image with bounding boxes saved at: {output_path}")
