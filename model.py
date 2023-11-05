# Importing required libs
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
from PIL import Image
from keras.preprocessing import image
import io

# Loading model
model = load_model("sign_classifie.h5")

def preprocess_img(img_path):
    img = image.load_img(io.BytesIO(img_path.read()), target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    # img = img / 255.0  # Normalize the image
    return img



# Predicting function
def predict_result(predict):
    pred = model.predict(predict)
    class_labels = ['A', 'ADD', 'B', 'Bent', 'Between', 'Blind', 'Bottle', 'Brain', 'Bowl', 'C', 'Bud', 'Chest', 'Claw', 'Coolie', 'Cough', 'D', 'Devil', 'Doctor', 'Cow', 'Afraid', 'E', 'East', 'Eight', 'Elbow', 'Evening', 'Eye', 'F', 'Faith', 'Fat', 'Feel', 'Fever', 'Few', 'First', 'Five', 'Food', 'Four', 'G', 'Good', 'Gun', 'Hair', 'Hand', 'Head', 'Hear', 'I', 'Jain', 'K', 'King', 'L', 'Leprosy', 'Love', 'M', 'Me', 'N', 'Nine', 'Nose', 'Nurse', 'O', 'Oath', 'One', 'Open', 'Owl', 'P', 'Police', 'Pray', 'Promise', 'Q', 'R', 'S', 'Seven', 'Shirt', 'Shoulder', 'Sick', 'Six', 'Skin', 'Sleep', 'Soldier', 'Stand', 'Strong', 'Sunday', 'T', 'Telephone', 'Ten', 'Thorn', 'Three', 'Tongue', 'Thumbs_up', 'Trouble', 'Two', 'U', 'V', 'W', 'Word', 'You', 'White', 'X', 'Zero', 'Z', 'Water', 'Wedding', 'West']
    predicted_class = class_labels[np.argmax(pred)]
    return predicted_class
