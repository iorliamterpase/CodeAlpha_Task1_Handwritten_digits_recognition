from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)
model = load_model('handwritten_model.h5')

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  
    image = np.array(image)
    image = image / 255.0 
    image = image.reshape(1, 28, 28, 1)  # CNN expects 4D input (batch, height, width, channels)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return render_template('index.html', prediction="No image uploaded!")

    file = request.files['image']
    try:
        image = Image.open(file.stream)
    except Exception:
        return render_template('index.html', prediction="Invalid image file!")

    processed = preprocess_image(image)
    prediction = model.predict(processed)
    predicted_digit = int(np.argmax(prediction))

    return render_template('index.html', prediction=predicted_digit)

if __name__ == '__main__':
    app.run(debug=True)
