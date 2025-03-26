from flask import Flask, render_template, request
import os
from model import DeepFakeDetector

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'

# Create folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Load DeepFake Detector Model
detector = DeepFakeDetector()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error="No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error="No selected file!")

    # Save Image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Predict DeepFake Probability
    probability = detector.predict(file_path)

    # Highlight Fake Regions
    processed_image = detector.highlight_fake_regions(file_path)

    result_text = "Fake" if probability > 0.5 else "Real"

    return render_template('result.html', 
                           image=file.filename, 
                           processed_image=file.filename, 
                           probability=probability * 100, 
                           result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
