# Import necessary libraries

from fastapi import FastAPI, File, UploadFile

from torch import unsqueeze
from torch import nn
from torch import no_grad
from torch import load
from torch import argmax

from torchvision import transforms, models

from PIL import Image

# Load trained model 

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 9)
model.load_state_dict(load('ft_model.pt'))
model.eval()

# Define transforms for data normalisation

data_transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Define dictionary to convert numerical predictions to text predictions

labels_map = {0: 'Cardboard', 1: 'Food Organics', 2: 'Glass', 3: 'Metal', 4: 'Miscellaneous Trash', 5: 'Paper', 6: 'Plastic', 7: 'Textile Trash', 8: 'Vegetation'}

# Define the app
app = FastAPI()

# Define root endpoint
@app.get('/')
def read_root():
    return {'message': 'A waste classification API'}

# Define the prediction endpoint
@app.post('/predict')
def pred(file: UploadFile = File(...)):
    x = data_transforms(Image.open(file.file))
    with no_grad():
        return {'prediction': labels_map[int(argmax(nn.functional.softmax(model(x.unsqueeze(0)), dim = 1)))]}