import streamlit as st
from CNN_model import ConvNet
from CNN_model import predict
from CNN_model import classes
import torch
from torchvision import transforms
from PIL import Image

model = ConvNet()
state_dict = torch.load("CONV_NET_v2.pth",map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

st.title("LeNet Architecture Trained on CIFAR-10 (PyTorch Demo)")
st.write("This model has been trained on the CIFAR-10 dataset and can classify images into 10 categories:")
st.write("Which includes: airplane, automobile, bird, cat, deer,dog, frog, horse, ship, truck")
st.write("*This model is prone to errors")

data = st.file_uploader("Upload an image",["jpeg","jpg","png"])
if data:
    st.image(data)
    prediction = predict(model,data,classes)
    st.write(f"# Model Prediction: {prediction}")
st.divider()


st.write("### The following image illustrates the architecture of the LeNet")
img_path = "resources/LeNet_architecture.png"
img = Image.open(img_path)
st.image(img)
st.divider()


img_path = "resources/code.png"
st.write("### The following code is the my implementation of the LeNet in pytorch")
img = Image.open(img_path)
st.image(img)


