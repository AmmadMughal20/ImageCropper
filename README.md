# Object Detection Web App

This is a simple Flask web application for object detection using the YOLO (You Only Look Once) model. Users can upload an image, and the application will use a pre-trained YOLO model to detect and highlight objects in the image. Additionally, the application provides the option to save the cropped region of the most confidently detected object.

## Project Structure

The project is organized into the following files and directories:

1. **app.py**: Contains the Flask web application code.
2. **static/**:
    - **best.pt**: Pre-trained YOLO model weights.
    - **images/**: Directory to store uploaded images.
        - *output/*: Directory to store prediction output images.
    - **styles/**: Directory for CSS stylesheets.
3. **templates/**:
    - **index.html**: HTML template for the main page.
    - **result.html**: HTML template for the result page.

## How to Run

1. Clone the repository:

    ```bash
    git clone https://github.com/AmmadMughal20/ImageCropper.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Flask application:

    ```bash
    python app.py
    ```

4. Open a web browser and navigate to [http://localhost:5000](http://localhost:5000) to use the application.

## Usage

1. Upload an image using the provided form on the main page.
2. Click the "Detect Objects" button to initiate object detection.
3. View the result on the result page, which includes the original image with bounding boxes around detected objects and the cropped region of the most confidently detected object.

## Additional Notes

- The YOLO model used in this project is loaded from the `best.pt` file.
- Detected objects are highlighted with bounding boxes, and the class name and confidence level are displayed.
- The cropped image of the most confidently detected object is saved in the `images/output/predict/` directory.

Feel free to customize and expand this project based on your requirements!
