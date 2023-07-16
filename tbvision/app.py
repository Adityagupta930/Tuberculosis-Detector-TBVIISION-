# app.py
import os
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__, template_folder='template')

# Set the path where the uploaded images will be stored permanently
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def Detector():
    return render_template('Detector.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return redirect(url_for('Detector'))
    
    image = request.files['image']
    
    if image.filename == '':
        return redirect(url_for('Detector'))

    if image:
        # Save the uploaded image to the specified upload folder
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
        # Optionally, you can perform any additional processing here
        # (e.g., use a machine learning model for image recognition, etc.)

        # Redirect to a success page or display the uploaded image
        return redirect(url_for('success'))

@app.route('/success')
def success():
    return "<h1>Image uploaded successfully!</h1>"

if __name__ == '__main__':
    app.run(debug=True)
