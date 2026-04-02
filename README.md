# 🎨 AI Product Ad Composer Studio

**Transform raw products into professional, branded advertisements in seconds.**

An intelligent, full-stack marketing tool that uses **Generative AI** (FLUX.1 & Qwen) and **Machine Learning** to create persona-targeted ad copy and ultra-realistic visuals from real ecommerce data.

---

## 🌟 Key Features

### 🛠️ Hybrid AI Engine
- **✨ Cloud API (High-End)**: Uses a powerful **Qwen-2.5 7B** model for creative, high-fidelity prompt generation. Fast, light, and perfect for deployment.
- **🤖 Local LLM (Downloaded)**: Support for local **Qwen-2.5 0.5B** inference for offline use and maximum privacy.

### 🖼️ Professional Visuals
- **High-Fidelity Images**: Powered by **FLUX.1 Schnell** for photorealistic commercial photography.
- **Branding Overlay**: Automatically applies a semi-transparent branding bar with your **Brand Name** and **Slogan** to the final image.
- **🔄 Regenerate**: Don't like a visual? Regenerate a new one with a single click using the same prompt.

### 📈 Smart Audience Prediction
- **AI-Powered Targeting**: Uses a trained **Random Forest/Linear Classifier** to automatically predict the best demographic (Teenagers, Professionals, Seniors) for any chosen product from the dataset.

### 📤 Seamless Export & Share
- **📥 Download Image**: Save high-resolution JPEGs with branding overlay instantly.
- **📋 Copy & Share Ad**: One-click to copy the ad text and app link to your clipboard, or use the native Share sheet (WhatsApp, Instagram, etc.).

---

## 🏗️ Project Architecture

```
project-ad-composer/
├── app.py                      # Main Streamlit Dashboard
├── requirements.txt            # Project dependencies
├── .env                        # Private API Keys (Hugging Face)
├── .gitignore                  # Prevents secrets & large files from Git
│
├── notebooks/
│   ├── Personalized_Ad_Composer.ipynb # Data Processing & ML Training
│   ├── audience_predictor.pkl         # Trained ML model
│   └── cleaned_product_data.csv       # Cleaned Dataset (Tracked)
│
└── flipkart_com-ecommerce_sample.csv  # Raw raw data (Excluded)
```

---

## ⚙️ Quick Start

### 1. Installation
```bash
git clone https://github.com/dhruvitabrainerhub/Product-Ad-composer.git
cd Product-Ad-composer
pip install -r requirements.txt
```

### 2. API Configuration
Create a `.env` file in the root directory and add your Hugging Face API key:
```bash
HUGGINGFACE_API_KEY=your_hf_token_here
```
> Get your free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 3. Run Locally
```bash
streamlit run app.py
```

---

## 🚀 Deployment (Streamlit Cloud)

1. Push your code to your GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. **Important**: Go to `Settings > Secrets` in the Streamlit dashboard and add your API key:
   ```toml
   HUGGINGFACE_API_KEY = "your_actual_key_here"
   ```

---

## 🛠️ Technology Stack

- **Frontend**: Streamlit (Premium UI)
- **Image Generation**: Hugging Face FLUX.1 (Schnell)
- **Text Generation**: Hybrid (Local Qwen / Cloud Qwen API)
- **Machine Learning**: Scikit-Learn / Joblib
- **Image Processing**: Pillow (PIL)
