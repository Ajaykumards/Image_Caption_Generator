import streamlit as st
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Optional: Install googletrans with pip install googletrans==4.0.0-rc1
try:
    from googletrans import Translator as GoogleTranslator
    googletrans_available = True
except ImportError:
    googletrans_available = False

st.title("🖼️ Image Caption Generator with Translation")
st.markdown("Generate captions using BLIP model and translate to Indian languages.")

option = st.radio("Choose image input method:", ("Upload Image", "Image URL"))

image = None
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

if image:
    with st.spinner("Generating caption..."):
        pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        result = pipe(image)
        caption = result[0]['generated_text']
        st.success("Caption Generated:")
        st.write(f"📝 **{caption}**")

    languages = {
        "English": "en",
        "Hindi": "hi",
        "Bengali": "bn",
        "Tamil": "ta",
        "Telugu": "te",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Punjabi": "pa",
        "Odia": "or",
        "Kannada": "kn",       # Not supported in Helsinki-NLP
        "Malayalam": "ml"      # Not supported in Helsinki-NLP
    }

    lang_choice = st.selectbox("Select language for translation:", list(languages.keys()))

    if lang_choice:
        target_lang_code = languages[lang_choice]
        if target_lang_code == "en":
            st.info("Caption is already in English.")
        else:
            # Helsinki-NLP supported languages only:
            hf_supported = {"hi","bn","ta","te","mr","gu","pa","or"}

            if target_lang_code in hf_supported:
                model_name = f"Helsinki-NLP/opus-mt-en-{target_lang_code}"
                with st.spinner(f"Translating caption to {lang_choice} using Hugging Face..."):
                    try:
                        translator = pipeline("translation_en_to_xx", model=model_name)
                        translation = translator(caption)[0]['translation_text']
                        st.success(f"Translated Caption ({lang_choice}):")
                        st.write(f"📝 **{translation}**")
                    except Exception as e:
                        st.error(f"Translation failed: {e}")

            else:
                # Fallback to googletrans for unsupported languages
                if not googletrans_available:
                    st.error("Googletrans package not installed. Please run `pip install googletrans==4.0.0-rc1`")
                else:
                    with st.spinner(f"Translating caption to {lang_choice} using Google Translate..."):
                        try:
                            gt = GoogleTranslator()
                            translated = gt.translate(caption, dest=target_lang_code)
                            st.success(f"Translated Caption ({lang_choice}):")
                            st.write(f"📝 **{translated.text}**")
                        except Exception as e:
                            st.error(f"Google Translate failed: {e}")
