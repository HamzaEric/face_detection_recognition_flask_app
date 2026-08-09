import io
import base64

from flask import Flask, render_template, request
from PIL import Image

# Import our face recognition code
from face_recognition import add_labels_to_image

# Starts Flask
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("upload.html")


@app.route("/recognize", methods=["POST"])
def process_image():
    # Display an error message if no image found
    if "image" not in request.files:
        return "No image provided", 400

    # Get the file sent along with the request
    file = request.files["image"]

    # Video also shows up as an image
    # we want to reject those as well
    if not file.mimetype.startswith("image/"):
        return "Image format not recognized", 400

    image_data = file.stream

    # Run our face recognition code!
    img_out = run_face_recognition(image_data)

    if img_out == Ellipsis:
        return "Image processing not enabled", 200
    else:
        # Convert the matplotlib figure to a byte buffer
        out_stream = matplotlib_to_bytes(img_out)

        # Encode the bytes into a base64 string
        encoded_img = base64.b64encode(out_stream.getvalue()).decode('utf-8')

        # Pass the base64 string to a new results template
        return render_template("result.html", image_data=encoded_img)


def run_face_recognition(image_data):
    # Open image_data with PIL
    input_image = Image.open(image_data)

    # Run our function on the PIL image
    img_out = add_labels_to_image(input_image)

    return img_out


def matplotlib_to_bytes(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, format="jpg", bbox_inches="tight")
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    app.run(debug=True)