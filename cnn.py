import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as mod
import torchvision.transforms as trans
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 

st.set_page_config(page_title="CNN Visualizer", page_icon="🧙🏽‍♂️", layout="wide")
st.title("VISUALIZER HEADING")

# Load pretrained model
model = mod.resnet18(pretrained=True)
model.eval()


# Image transforms
transform = trans.Compose([
    trans.Resize((224, 224)),
    trans.ToTensor(),
    trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Upload image
uploaded = st.file_uploader("UPLOAD AN IMAGE", type=["jpg", "png", "jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)  # avoid shape mismatch

    # First conv layer feature maps
    with torch.no_grad():
        first_layer = model.conv1(input_tensor)

    st.subheader("Feature Maps")

    # Show first 10 fmaps bs
    for i in range(5):
        fmap = first_layer[0, i].cpu().numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-5)  # normalize to [0,1] (fixed)
        st.image(fmap, caption=f"Feature map {i+1}", use_column_width=True)

else:
    st.write("Upload an image to get started")
