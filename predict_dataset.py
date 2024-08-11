import tensorflow as tf
import numpy as np
from PIL import Image
import tensorflow as tf
model = tf.keras.models.load_model('cats_dogs_model')

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((32, 32))
    
    # Convert to RGB in case the image has an alpha channel
    img = img.convert('RGB')
    
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# List of class names
class_names = ['cats','dogs']

# Predict the class
image_path_1 = 'predictions/cat_or_dog_1.jpg'
image_path_2 = 'predictions/cat_or_dog_2.jpg'

def predict(image_path):
    input_image = preprocess_image(image_path)
    predictions = model.predict(input_image)

    predicted_class = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class]

    print("Predicted class:", predicted_class_name)
    
predict(image_path_2)