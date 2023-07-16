from flask import Flask, render_template, request
import torch

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('Detector.html')

@app.route('/process', methods=['POST'])
def process():
    # Handle the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = "path/to/save/file"
        uploaded_file.save(file_path)

        # Process the file using your model
        result = process_file(file_path)

        # Return the result
        return result
    else:
        return "No file uploaded"
    
def process_file(file_path):
    # Load the model from .pth or .onnx file
    model = torch.onnx.load(tuberModel.onnx)  # If using .pth file
    # model = torch.onnx.load(file_path)  # If using .onnx file

    # Process the file using the model
    # Perform any necessary preprocessing, inference, or postprocessing

    # Return the processed result
    result = "Processed result"
    return result

if __name__ == '__main__':
    app.run()