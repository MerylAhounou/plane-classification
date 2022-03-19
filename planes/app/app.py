import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from PIL import Image

IMAGE_WIDTH = 128
IMAGE_HEIGHT = IMAGE_WIDTH
IMAGE_DEPTH = 3


def load_image(path):
    """Load an image as numpy array
    """
    return plt.imread(path)


def predict_image(path, model):
    """Predict plane identification from image.
    
    Parameters
    ----------
    path (Path): Path to image to identify
    model (keras.models): Keras model to be used for prediction
    Returns
    -------
    Predicted class
    """
    names = pd.read_csv('../models/classes_names.txt', names=['Names'])
    image_resized = [np.array(Image.open(path).resize((IMAGE_WIDTH, IMAGE_HEIGHT)))]
    prediction_vector = model.predict(np.array(image_resized))
    predicted_classes = np.argmax(prediction_vector)
    name_classes = names['Names'][predicted_classes]
    return name_classes

def load_model(path):
    """Load tf/Keras model for prediction
    """
    return tf.keras.models.load_model(path)


model = load_model("../models/my_model.h5")
model.summary()

st.title("Identifation d'avion")

uploaded_files = st.file_uploader("Charger une image d'avion") #, accept_multiple_files=True)
st.write(uploaded_files)

if uploaded_files:
    loaded_image = load_image(uploaded_files)
    st.image(loaded_image)
    
predict_btn = st.button('Identifier', disabled=(uploaded_files is None))
if predict_btn:
    prediction = predict_image(uploaded_files, model)
    st.write(f"C'est un: {prediction}")
    # Exemple si les f-strings ne sont pas dispo.
    # st.write("C'est un: {}".format(prediction))
