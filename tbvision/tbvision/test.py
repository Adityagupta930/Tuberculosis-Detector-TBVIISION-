from flask import Flask,request
import torch
import torch.nn as nn
from torchvision import transforms

app = Flask(__name__)

def load_model(model_path):
    model=torch.model('01_tuberculosis_model.pth')
    model.eval()
    return model

def make_prediction(model,input_tensor):
    with torch.no_gard():
        output = model(input_tensor)
        return output
    


@app.route('/upload',method=['POST'])
def upload():
    file=request.files['file']
    input_tensor=preprocess(file)
    model=load_model('01_tuberculosis_model.pth')
    prediction = make_prediction(model,input_tensor)
    return 'prediction '+str(prediction)


def preprocess(file):
    # Perform any necessary preprocessing steps on the file or input data
    # Convert the file to an image tensor or process the data accordingly
    # For example, if you are working with image data, you can use torchvision.transforms to preprocess the image

    # Example preprocessing steps for image data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to a specific size
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
    ])

    # Open the uploaded file and apply the transformations
    image = Image.open(file)
    processed_image = transform(image)

    # Reshape the tensor to match the expected input shape of your model
    input_tensor = processed_image.unsqueeze(0)

    return input_tensor

if __name__ == '__main__':
    app.run()
