import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image

# Set the page layout and title
st.set_page_config(page_title="Veg/Fruit Image Classifier", page_icon="🍏", layout="centered")

# Inject custom CSS to style the page
st.markdown("""
    <style>
    /* Page background */
    .stApp {
        background-image: linear-gradient(to right, #f0f4c3, #c8e6c9);
        color: #343a40;
    }
    
    /* Header styling */
    .stHeader h1 {
        font-family: 'Trebuchet MS', sans-serif;
        color: #2e7d32;
        text-align: center;
        font-size: 3em;
        margin-bottom: 10px;
    }

    /* File uploader styling */
    .stFileUploader {
        text-align: center;
        border-radius: 10px;
        background-color: #aed581;
        color: #ffffff;
        font-weight: bold;
        padding: 10px;
    }
    
    /* Image display styling */
    img {
        border-radius: 10px;
        margin-top: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Result styling */
    .stMarkdown h1 {
        font-family: 'Verdana', sans-serif;
        color: #2e7d32;
        font-size: 1.8em;
        text-align: center;
    }

    /* Confidence score styling */
    .confidence {
        font-size: 1.5em;
        color: #f57c00;
        font-weight: bold;
        text-align: center;
    }

    </style>
""", unsafe_allow_html=True)

# Header
st.header('Image Classification Model 🍏')

# Load the model
model = load_model(r'B:/Harsha/Praxis/Image classification/Code/Image_classify.keras')

# List of categories for classification
data_cat = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
    'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic',
    'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
    'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato',
    'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon'
]

# Define image height and width
img_height = 180
img_width = 180

# Create a file uploader for the image
uploaded_file = st.file_uploader("Upload an image of a fruit or vegetable...", type=["jpg", "jpeg", "png"])

# Proceed if a file is uploaded
if uploaded_file is not None:
    # Open and preprocess the image
    image = Image.open(uploaded_file)
    img_arr = np.array(image.resize((img_height, img_width)))  # Resize and convert to array
    img_bat = np.expand_dims(img_arr, axis=0)  # Add batch dimension

    # Make prediction
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict[0])

    # Display the image and prediction results (reduced image size)
    st.image(image, caption='Uploaded Image', width=300)  # Reduced width to 300px
    
    st.markdown(f"**Veg/Fruit in image is: {data_cat[np.argmax(score)]}**")
    st.markdown(f"<div class='confidence'>Confidence: {np.max(score) * 100:.2f}%</div>", unsafe_allow_html=True)
else:
    st.write("Please upload an image to classify.")
