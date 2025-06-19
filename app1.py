import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Title
st.title("🖼️ Image Caption Generator")
st.markdown("Generate captions using BLIP model from Hugging Face 🤗")

# Option to upload or use URL
option = st.radio("Choose image input method:", ("Upload Image", "Image URL"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

elif option == "Image URL":
    image_url = st.text_input("Enter image URL:")
    if image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="Image from URL", use_column_width=True)
        except:
            st.error("Failed to load image. Please check the URL.")

# Load model and generate caption
if "image" in locals():
    with st.spinner("Generating caption..."):
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        result = pipe(image)
        st.success("Caption Generated:")
        st.write(f"📝 **{result[0]['generated_text']}**")
