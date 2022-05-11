from numpy import imag
import streamlit as st
from urllib.request import urlretrieve

st.title("Savory Unsavory Classifier😎😎")
rslt = st.subheader("")
loading_text = st.text("Loading...")

def load_module(loading_text):
  loading_text.text('Loading Module ✊')
  from fastai.learner import load_learner
  loading_text.text("Loading Module ✅")
  global load_learner


def load_model(loading_text):
  loading_text.text("Loading Model 🛑")
  from urllib.request import urlretrieve
  from pathlib import Path
  urlretrieve("https://github.com/lkarjun/savory-unsavory-classifier/blob/master/Models/model-2022-05-10%2006_45_27.074624.pkl?raw=true", 
              "model.pkl")
  loading_text.text("Loading Model ✅")
  return load_learner("model.pkl")

def load_image(loading_text, image):
  loading_text.text("Loading image ✊")
  from fastai.vision.core import Image, tensor
  data = tensor(Image.open(image))
  loading_text.text("Loading image ✅")
  return data


load_module(loading_text)

image_file = st.camera_input("")
loading_text.text("Please take a picture 📸")
if image_file:
    image = load_image(loading_text, image_file)
    learn = load_model(loading_text)
    prediction = learn.predict(image)
    rslt.subheader(f"You're an {prediction[0]}")