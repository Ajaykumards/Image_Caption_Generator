### 💻 Languages and Tools
<div align="left">
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" height="30" alt="python logo" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/streamlit/streamlit-original.svg" height="30" alt="streamlit logo" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pytorch/pytorch-original.svg" height="30" alt="pytorch logo" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" height="30" alt="tensorflow logo" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/git/git-original.svg" height="30" alt="git logo" />
  <img width="12" />
  <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30" alt="github logo" />
</div>

---

### 📈 GitHub Contribution Graph

![GitHub Contribution Graph](https://ghchart.rshah.org/ajaykumards)

---

# 🖼️ Image Caption Generator with Translation

Generate captions for images using the BLIP model and translate them into various Indian languages.

---

## Features
- Upload an image or provide an image URL.
- Supports translation via Hugging Face Helsinki-NLP models.

---

## Preview

## 🔍 Preview - Image Upload Section

![Upload Preview](https://github.com/ajaykumards/image-caption-generator/raw/main/screenshots/img_url_uploader.png)

![Translation Result Preview](https://yourdomain.com/images/translation_preview.png)


---

## How to Run Locally

### Prerequisites
- Python 3.7 or higher
- Streamlit
- Transformers
- PIL (Pillow)
- Requests
- Optionally: googletrans (`pip install googletrans==4.0.0-rc1`) for languages not supported by Helsinki-NLP

### Installation

```bash
pip install streamlit transformers pillow requests
pip install googletrans==4.0.0-rc1  # Optional, for some Indian languages
