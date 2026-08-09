# Face Detection & Recognition Flask App

A lightweight, production-ready computer vision web application built with **Python**, **Flask**, and **PyTorch**, designed to perform closed-set facial recognition and identity verification against a targeted database of specific reference individuals.

---

## Key Features

* **High-Precision Face Detection (MTCNN):** Leverages Multi-task Cascaded Convolutional Networks to detect, crop, and align faces from raw image uploads.
* **Deep Feature Extraction (InceptionResnetV1):** Uses pre-trained deep architecture (`facenet-pytorch`) to extract 512-dimensional facial feature embeddings.
* **Closed-Set Identity Matching:** Performs distance vector calculations against pre-computed reference embeddings (`embeddings.pt`) for targeted identity verification.
* **Production-Grade Cloud Deployment:** Hosted live on **Azure App Service** (Linux B1 SKU) with Gunicorn and custom timeout configurations to prevent out-of-memory (OOM) worker failures.

---

## Tech Stack

* **Language:** Python 3.11

* **Web Framework:** Flask, Gunicorn

* **Machine Learning:** PyTorch, facenet-pytorch, torchvision, PIL

* **Deployment & Hosting:** Microsoft Azure App Service (Linux B1 Tier), GitHub Actions (CI/CD)

## Project Structure

```text
├── Face embeddings/          # Reference embedding datasets
├── flask_app/                # Main application package
│   ├── static/               # Static assets (CSS, images)
│   ├── templates/            # Front-end HTML templates
│   ├── embeddings.pt         # Pre-computed target facial embeddings
│   ├── face_recognition.py   # Core AI engine & inference pipelines
│   └── flask_app.py          # Flask app entrypoint & route handlers
└── Notebooks/                # Research & development notebooks
    ├── face_detection_using_mtcnn.ipynb
    ├── face_recognition_using_Inception_Resnet (1).ipynb
    └── Fetch_&_Data_Exploration.ipynb
