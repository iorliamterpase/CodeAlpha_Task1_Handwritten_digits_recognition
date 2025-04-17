# model.py
import tensorflow as tf

class ModelLoader:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, img):
        img = img.reshape(-1).astype('float32') / 255.0  # Flatten and normalize to (784,)
        img = img.reshape(1, 784)  # Reshape to (1, 784)
        predictions = self.model.predict(img)
        return predictions.argmax(axis=1)[0]  # Get the predicted class