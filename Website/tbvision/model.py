import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the trained model
model = torch.load('tuberModel.onnx')
model.eval()

# Check if a GPU is available and move the model to the GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define the transformation sequence
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the input image
image_path = 'image3.jpg'
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)

# Move the input tensor to the same device as the model
input_batch = input_batch.to(device)

# Make predictions
with torch.no_grad():
    output = model(input_batch)

# Get the predicted class probabilities
probabilities = torch.softmax(output, dim=1)[0]
tb_probability = probabilities[1].item()  # Probability of TB class

# Define a threshold for classification
threshold = 0.5

# Classify based on the threshold
if tb_probability >= threshold:
    prediction = 'TB'
else:
    prediction = 'No TB'

# Print the prediction and probability
print("Prediction:", prediction)
print("TB Probability:",tb_probability)