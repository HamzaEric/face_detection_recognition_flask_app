from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")


# ============================================================
# LOAD MODELS
# ============================================================

mtcnn = MTCNN(
    image_size=160,
    keep_all=True,
    min_face_size=40,
    device=device
)

resnet = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

embedding_data = torch.load(
    "embeddings.pt",
    map_location=device
)


# ============================================================
# LOCATE FACES
# ============================================================

def locate_faces(image):

    # MTCNN performs detection and returns the
    # cropped faces and probabilities.
    cropped_images, probs = mtcnn(
        image,
        return_prob=True
    )

    # If no face was detected
    if cropped_images is None or probs is None:
        return []

    # Get bounding boxes separately.
    # This is still another detection internally,
    # but we need the coordinates for drawing.
    boxes, _ = mtcnn.detect(image)

    if boxes is None:
        return []

    return list(zip(boxes, probs, cropped_images))


# ============================================================
# DETERMINE NAME + DISTANCE
# ============================================================

def determine_name_dist(cropped_image, threshold=0.9):

    # Generate embedding without creating
    # a computational graph.
    with torch.no_grad():

        emb = resnet(
            cropped_image
            .unsqueeze(0)
            .to(device)
        )

    distances = []

    for known_emb, name in embedding_data:

        known_emb = known_emb.to(device)

        dist = torch.dist(
            emb,
            known_emb
        ).item()

        distances.append(
            (dist, name)
        )

    # Find closest embedding
    dist, closest = min(
        distances,
        key=lambda x: x[0]
    )

    if dist < threshold:
        name = closest
    else:
        name = "Undetected"

    return name, dist


# ============================================================
# LABEL FACE
# ============================================================

def label_face(name, dist, box, axis):

    rect = plt.Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        fill=False,
        color="blue"
    )

    axis.add_patch(rect)

    if name == "Undetected":
        color = "red"
    else:
        color = "blue"

    label = f"{name} {dist:.2f}"

    axis.text(
        box[0],
        box[1],
        label,
        fontsize="large",
        color=color
    )


# ============================================================
# ADD LABELS TO IMAGE
# ============================================================

def add_labels_to_image(image):

    width, height = image.width, image.height

    dpi = 96

    fig = plt.figure(
        figsize=(
            width / dpi,
            height / dpi
        ),
        dpi=dpi
    )

    axis = fig.subplots()

    axis.imshow(image)
    axis.axis("off")

    # Detect faces
    faces = locate_faces(image)

    for box, prob, cropped in faces:

        if prob is None or prob < 0.90:
            continue

        name, dist = determine_name_dist(
            cropped
        )

        label_face(
            name,
            dist,
            box,
            axis
        )

    return fig
