from flask import Flask, render_template, request, url_for
import subprocess
import os
import random
import string
import shutil

app = Flask(__name__)


@app.route("/")
def index():
    images_directory = 'app/static/images'
    image_filenames = [filename for filename in os.listdir(images_directory) if filename.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Generate URLs for the images using url_for
    image_urls = [url_for('static', filename=f'images/{filename}') for filename in image_filenames]

    return render_template("index.html", image_urls=image_urls)


@app.route("/detect", methods=["POST"])
def detect():
    # Specify the path to the trained YOLOv8 weights file
    weights_path = "app/static/best.pt"

    # Get the image from the request
    image = request.files["image"].read()

    image_filename = generate_random_filename()
    image_path = os.path.join("app/static/images", image_filename)

    # Ensure the 'images/' directory exists
    os.makedirs("app/static/images", exist_ok=True)

    output_directory = 'app/static/images/output'
    os.makedirs(output_directory, exist_ok=True)

    source_directory = 'runs/detect/predict'
    destination_directory = os.path.join(output_directory, 'predict')

    render_directory = 'images/output/predict'
    with open(image_path, "wb") as temp_image:
        temp_image.write(image)
    # Perform object detection using the saved weights and image
    subprocess.run(
        [
            "yolo",
            "task=detect",
            "mode=predict",
            f"model={weights_path}",
            f"source={image_path}",
        ]
    )

    # Move the files directly to the destination directory
    for filename in os.listdir(source_directory):
        source_path = os.path.join(source_directory, filename)
        destination_path = os.path.join(destination_directory, filename)
        shutil.move(source_path, destination_path)
    
    shutil.rmtree(source_directory)

    return render_template(
        "result.html", image_path=render_directory + "/" + image_filename
    )


def generate_random_filename():
    # Generate a random string of letters and digits
    random_chars = "".join(random.choices(string.ascii_letters + string.digits, k=8))

    # Combine the random string and timestamp to create a unique filename
    filename = f"{random_chars}.jpg"

    return filename
