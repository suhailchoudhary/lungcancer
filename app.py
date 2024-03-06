from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the saved model
model_path = 'cancer_model_lung.h5'  # Replace with the actual path to your saved model file
model = keras.models.load_model(model_path)

# Assuming you have the classes list
classes = ['lung_aca', 'lung_n', 'lung_scc']

@app.route('/')
def home():
    return render_template('anotherindex.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Check if the file is an allowed extension
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file extension'})

    # Read and preprocess the image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))  # Resize to match the model's input size
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 3))

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    predicted_class_label = classes[predicted_class]

    response = {
        'predicted_class': predicted_class_label,
        'confidence': float(confidence),
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
