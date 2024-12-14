This repository contains the files used in building a simple waste classification API. The API passes images of waste/trash to a deep neural network which predicts the waste category.

We first fine-tuned a ResNet model in the waste_model.ipynb Jupyter notebook using PyTorch (torchvision), and using the dataset [RealWaste](https://github.com/sam-single/realwaste/tree/main). We then set up a simple prediction API using FastAPI in api/main.py. To handle dependencies and allow deployment, we then built a Docker image using api/Dockerfile.txt which can be run as a container.
