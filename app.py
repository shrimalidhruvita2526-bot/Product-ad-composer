"""
AI Product Ad Composer — Main Application
==========================================
Template Engine for ad copy (headline/description/slogan).
Local LLM (Qwen2.5-0.5B-Instruct) on your NVIDIA GPU for image prompt generation.
Hugging Face FLUX.1 API for high-fidelity image generation.
"""

import base64
import json
import logging
import os
from pathlib import Path
import pickle

import io
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageColor
from dotenv import load_dotenv
import random

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Try loading heavy ML libraries (Hybrid Fallback) ---
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    ML_AVAILABLE = True
except ImportError:
    logger.warning("Local ML libraries (torch/transformers) not found. App will use Cloud API mode.")
    ML_AVAILABLE = False

# ---------------------------------------------------------------------------
# Environment & API Keys
# ---------------------------------------------------------------------------
load_dotenv(override=True)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# Base directory — resolved from *this* file's location
BASE_DIR = Path(__file__).resolve().parent

# Local LLM model to use for image prompt generation
LOCAL_LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Simple cache to avoid regenerating prompt for same product/demographic
_prompt_cache: dict = {}

# Cloud AI Text Inference URL (Qwen-7B is more creative than local 0.5B!)
HF_TEXT_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"

# ---------------------------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="AI Ad Composer Studio", layout="wide", page_icon="🎨")

# ---------------------------------------------------------------------------
# Custom CSS — Premium Look
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4F46E5;
        color: white;
        font-weight: bold;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .ad-card {
        padding: 24px;
        border-radius: 12px;
        background-color: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        color: #111827 !important; /* Force dark text for readability in Dark Mode */
    }
    .ad-card h3 {
        color: #4F46E5 !important;
        margin-top: 0;
    }
    .ad-card p {
        color: #374151 !important;
        line-height: 1.6;
    }
    .ad-card hr {
        margin: 1rem 0;
        border-color: #f3f4f6;
    }
    .action-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid #e5e7eb;
        background: white;
        color: #374151;
        text-decoration: none;
        gap: 8px;
    }
    .action-button:hover {
        background-color: #f9fafb;
        border-color: #d1d5db;
        transform: translateY(-1px);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Core Configuration
# ---------------------------------------------------------------------------
DEMOGRAPHIC_STYLES = {
    "Teenagers": {
        "aesthetic": "Trendy, energetic, bold colors, vibrant pop-art style",
        "tone": "Playful, popular, high energy",
        "keywords": ["fresh", "popular", "must-have", "epic", "trendy"]
    },
    "Professionals": {
        "aesthetic": "Minimalistic, premium, sophisticated, sleek professional studio lighting",
        "tone": "Confident, concise, authoritative",
        "keywords": ["efficiency", "premium", "success", "reliability", "sleek"]
    },
    "Seniors": {
        "aesthetic": "Calm, trustworthy, warm lighting, simple and clear composition",
        "tone": "Gentle, reassuring, simple, respectful",
        "keywords": ["comfort", "trusted", "easy", "quality", "classic"]
    }
}

HF_IMAGE_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"

# ---------------------------------------------------------------------------
# Local LLM — Load Once & Cache (Qwen2.5-0.5B on GPU)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="🤖 Loading Local LLM (Qwen2.5-0.5B) — first time only...")
def get_local_llm_pipeline():
    """
    Load Qwen2.5-0.5B-Instruct model into a text-generation pipeline.
    """
    if not ML_AVAILABLE:
        logger.warning("Local ML libraries missing — skipping model load.")
        return None, "cpu"

    logger.info("Loading local LLM: %s", LOCAL_LLM_MODEL)
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using device: %s", device)
        
        # Load logic...
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL)
        model = AutoModelForCausalLM.from_pretrained(
            LOCAL_LLM_MODEL, 
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto"
        )
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return pipe, device
    except Exception as e:
        logger.error(f"Local model load failed: {e}")
        return None, "cpu"


# ---------------------------------------------------------------------------
# Business Logic — Ad Copy via Template Engine
# ---------------------------------------------------------------------------

def generate_ad_copy_template(product_name: str, category: str, demographic: str) -> dict:
    """
    Deterministic, instant ad copy generation using persona-matched templates.
    No API or GPU needed.
    """
    style = DEMOGRAPHIC_STYLES.get(demographic, DEMOGRAPHIC_STYLES["Professionals"])
    kw = style["keywords"]
    tone = style["tone"]

    templates = {
        "Teenagers": {
            "headline": f"{product_name} — Ekdum {kw[0].title()} aur {kw[3].title()}! 🔥",
            "description": (
                f"Apna {category.lower()} game level up karo {product_name} ke saath! "
                f"Ye hai market ka sabse {kw[2]} pick jo aapko baki sabse aage rakhega. "
                f"Aapke {kw[1]} friends bhi ise hi mang rahe hain."
            ),
            "slogan": f"Be Bold. Be {kw[4].title()}. Be You.",
        },
        "Professionals": {
            "headline": f"{product_name} — {kw[1].title()} Performance jo aapke kaam aaye",
            "description": (
                f"Apne {category.lower()} standards ko elevate kariye {product_name} ke saath. "
                f"Ye un professionals ke liye hai jo {kw[0]} aur {kw[3]} par compromise nahi karte. "
                f"Aapki success ab aur bhi sleek hogi."
            ),
            "slogan": f"{kw[1].title()} aur Precision ka Perfect Mix.",
        },
        "Seniors": {
            "headline": f"{product_name} — {kw[1].title()}, {kw[0].title()}, aur Sirf Aapke Liye",
            "description": (
                f"Ab experience kariye {product_name} ka asli {kw[3]}. "
                f"Ise banaya gaya hai simplicity aur {kw[0]} ko dhyan mein rakh kar. "
                f"Lakho logon ka bharosa aur behatrin {kw[1]} quality."
            ),
            "slogan": f"{kw[1].title()} Jis Par Aap Kar Sakein Bharosa.",
        },
    }

    result = templates.get(demographic, templates["Professionals"])
    logger.info("Template ad copy generated for demographic: %s", demographic)
    return result


# ---------------------------------------------------------------------------
# Business Logic — Image Prompt via Local LLM
# ---------------------------------------------------------------------------

def generate_image_prompt_api(api_key: str, prompt_input: str) -> str | None:
    """
    Call Hugging Face's Text Inference API to generate a creative image prompt.
    """
    headers = {"Authorization": f"Bearer {api_key.strip()}"}
    payload = {
        "inputs": f"<|im_start|>system\nYou are a world-class creative director and product photographer. Write ONE extremely detailed, photorealistic image prompt (max 60 words) for a high-end commercial ad. Focus on the authentic physical design of the product, its texture, and professional studio lighting. Output ONLY the prompt text.<|im_end|>\n<|im_start|>user\n{prompt_input}<|im_end|>\n<|im_start|>assistant\n",
        "parameters": {"max_new_tokens": 120, "temperature": 0.3}
    }
    try:
        response = requests.post(HF_TEXT_URL, headers=headers, json=payload, timeout=20)
        if response.status_code == 200:
            result = response.json()
            # Handle different HF response formats
            if isinstance(result, list) and len(result) > 0:
                text = result[0].get("generated_text", "")
                # Clean up Qwen chat format if it leaked
                if "assistant\n" in text:
                    text = text.split("assistant\n")[-1].strip()
                return text
        return None
    except Exception as e:
        logger.error(f"Cloud AI Prompt failed: {e}")
        return None

def generate_image_prompt_hybrid(
    product_name: str,
    demographic: str,
    slogan: str,
    brand_name: str,
    engine: str,
    llm_pipeline = None,
    api_key: str = None
) -> str:
    """
    Hybrid prompt generator - chooses between Local LLM and Cloud API.
    """
    cache_key = f"{product_name}|{demographic}|{brand_name}|{engine}"
    if cache_key in _prompt_cache:
        return _prompt_cache[cache_key]

    style = DEMOGRAPHIC_STYLES.get(demographic, DEMOGRAPHIC_STYLES["Professionals"])
    prompt_input = (
        f"Product: {product_name}. Brand: {brand_name}. Audience: {demographic}. "
        f"Style: {style['aesthetic']}. "
        f"Authentic product design, hyper-realistic, photorealistic commercial photography, 8k resolution, elegant studio lighting, sharp focus on product textures. "
        f"NO TEXT, NO TYPOGRAPHY, NO WORDS, NO SLOGANS, NO GIBBERISH inside the image."
    )

    prompt_text = None

    # Option 1: Cloud API (Fast & Advanced)
    if engine == "✨ Generate Fast (Use if loading)" and api_key:
        with st.spinner("☁️ AI working in the cloud..."):
            prompt_text = generate_image_prompt_api(api_key, prompt_input)

    # Option 2: Local LLM (Downloaded)
    elif engine == "🤖 Generate Local" and llm_pipeline:
        try:
            # Re-using the logic from the previous generate_image_prompt_llm
            system_msg = "You are a professional creative director. Write ONE extremely detailed, photorealistic image prompt for a commercial product advertisement. Focus ONLY on the visual design, lighting, and textures. Output ONLY the prompt text. NO TEXT INSIDE THE IMAGE."
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_input},
            ]
            output = llm_pipeline(messages, max_new_tokens=80, do_sample=True, temperature=0.3)
            generated = output[0]["generated_text"]
            prompt_text = generated[-1].get("content", "").strip() if isinstance(generated, list) else str(generated).strip()
        except: pass

    # Fallback if both fail or not selected
    if not prompt_text:
        prompt_text = (
            f"Hyper-realistic professional commercial photography of {product_name}, "
            f"authentic {style['aesthetic']} design, luxury studio lighting, high-end product shot, "
            f"8k resolution, extremely detailed, macro lens sharp focus."
        )

    _prompt_cache[cache_key] = prompt_text
    return prompt_text


# ---------------------------------------------------------------------------
# Business Logic — Image Generation via Hugging Face API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Business Logic — Image Branding Overlay
# ---------------------------------------------------------------------------

def add_branding_overlay(image: Image.Image, brand_name: str, slogan: str) -> Image.Image:
    """Adds a luxury-grade, semi-transparent bar at the bottom with centered Brand and Slogan."""
    import matplotlib
    import os
    
    # Clone image to avoid modifying original
    img = image.copy().convert("RGBA")
    width, height = img.size
    
    # Scaled bar height (Increased to 20% for much better visibility)
    bar_height = int(height * 0.20)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Darker semi-transparent black bar for maximum text contrast
    draw.rectangle([0, height - bar_height, width, height], fill=(0, 0, 0, 220))
    
    # Robust Font Strategy — Use Matplotlib Bundled Fonts (Always present on Streamlit)
    brand_size = int(bar_height * 0.45)    # Large brand name
    slogan_size = int(bar_height * 0.25)   # Very readable slogan
    
    font_path_bold = None
    font_path_reg = None
    
    try:
        mpl_font_base = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
        possible_bold = os.path.join(mpl_font_base, "DejaVuSans-Bold.ttf")
        possible_reg = os.path.join(mpl_font_base, "DejaVuSans.ttf")
        
        if os.path.exists(possible_bold):
            font_path_bold = possible_bold
        if os.path.exists(possible_reg):
            font_path_reg = possible_reg
    except:
        pass

    # Fallback paths if above search fails
    if not font_path_bold:
        fallback_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "arialbd.ttf"
        ]
        for p in fallback_paths:
            if os.path.exists(p):
                font_path_bold = p
                break

    try:
        if font_path_bold:
            brand_font = ImageFont.truetype(font_path_bold, brand_size)
        else:
            brand_font = ImageFont.load_default()
            
        if font_path_reg:
            slogan_font = ImageFont.truetype(font_path_reg, slogan_size)
        elif font_path_bold:
            slogan_font = ImageFont.truetype(font_path_bold, slogan_size)
        else:
            slogan_font = ImageFont.load_default()
    except:
        brand_font = ImageFont.load_default()
        slogan_font = ImageFont.load_default()

    # Draw Brand (Uppercase & bright white)
    brand_text = str(brand_name).upper()
    try:
        b_left, b_top, b_right, b_bottom = draw.textbbox((0, 0), brand_text, font=brand_font)
        brand_w, brand_h = b_right - b_left, b_bottom - b_top
    except AttributeError:
        brand_w, brand_h = draw.textsize(brand_text, font=brand_font)
    
    draw.text(
        ((width - brand_w) // 2, height - bar_height + (bar_height // 6)),
        brand_text,
        font=brand_font,
        fill=(255, 255, 255, 255)
    )
    
    # Draw Slogan (Centered under Brand)
    try:
        s_left, s_top, s_right, s_bottom = draw.textbbox((0, 0), slogan, font=slogan_font)
        slogan_w, slogan_h = s_right - s_left, s_bottom - s_top
    except AttributeError:
        slogan_w, slogan_h = draw.textsize(slogan, font=slogan_font)
    
    draw.text(
        ((width - slogan_w) // 2, height - bar_height + (bar_height // 1.6)),
        slogan,
        font=slogan_font,
        fill=(255, 255, 255, 230)
    )
    
    return Image.alpha_composite(img, overlay).convert("RGB")

def generate_hf_image(api_key: str, prompt: str) -> bytes | None:
    """
    Call Hugging Face's Image Inference API with multiple fallbacks.
    If the primary model (FLUX) is out of quota (402), it tries secondary models.
    """
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": prompt,
        "parameters": {"seed": random.randint(1, 1000000)}
    }

    # List of models to try in order of quality
    # (Display Name, API URL)
    endpoints = [
        ("FLUX.1 🔥", "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"),
        ("Stable Diffusion XL 🚀", "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"),
        ("SD v1.5 (Reliable) 🛡️", "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"),
    ]

    for name, url in endpoints:
        try:
            logger.info("Attempting image generation with %s at: %s", name, url)
            response = requests.post(url, headers=headers, json=payload, timeout=120)

            if response.status_code == 200:
                logger.info("Image generated successfully using %s", name)
                # Show which model was used in the UI for transparency
                st.toast(f"✅ Generated using {name}", icon="🎨")
                return response.content

            # Handle Quota / Credit errors (402/429) specifically
            if response.status_code in [402, 429]:
                st.warning(f"⚠️ {name} quota full or too busy. Trying fallback model...")
                continue
            
            error_data = response.json() if response.text.startswith("{") else {}
            error_msg = error_data.get("error", response.text)
            logger.warning("Failed at %s: %s — %s", name, response.status_code, error_msg)

        except Exception as e:
            logger.warning("Error connecting to %s: %s", name, e)

    st.error("🚨 All image generation models failed. Please check your API key or try again after 24 hours.")
    return None


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading product catalogue...")
def load_data() -> pd.DataFrame | None:
    """Load the cleaned product dataset from the project root."""
    cleaned_path = BASE_DIR / "cleaned_product_data.csv"
    if not cleaned_path.exists():
        cleaned_path = BASE_DIR / "notebooks" / "cleaned_product_data.csv"

    raw_path = BASE_DIR / "flipkart_com-ecommerce_sample.csv"

    if cleaned_path.exists():
        try:
            df = pd.read_csv(cleaned_path)
            required_cols = ["product_name", "description", "main_category"]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = ""
            df["product_name"] = df["product_name"].fillna("Unknown Product")
            logger.info("Loaded dataset: %d products", len(df))
            return df
        except (pd.errors.ParserError, OSError) as e:
            logger.error("Failed to parse %s: %s", cleaned_path, e)

    if raw_path.exists():
        try:
            logger.warning("Cleaned CSV not found. Falling back to raw Flipkart dataset.")
            return pd.read_csv(raw_path)
        except (pd.errors.ParserError, OSError) as e:
            logger.error("Failed to parse fallback CSV: %s", e)

    logger.error("No product dataset found.")
    return None


df = load_data()


@st.cache_resource(show_spinner=False)
def load_ml_model():
    """Load the trained audience predictor ML model."""
    model_path = BASE_DIR / "notebooks" / "audience_predictor.pkl"
    if model_path.exists():
        try:
            logger.info("Loading ML model from %s...", model_path)
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error("Failed to load ML model: %s", e)
    
    logger.warning("No audience predictor model found.")
    return None


ml_model = load_ml_model()

# Load the local LLM pipeline at startup (cached — only loads once)
llm_pipe, llm_device = get_local_llm_pipeline()

# ---------------------------------------------------------------------------
# Bulk Processing Helpers
# ---------------------------------------------------------------------------

def generate_bulk_template() -> bytes:
    """Creates a sample CSV template for bulk uploads."""
    template_df = pd.DataFrame([
        {
            "product_name": "Premium Leather Wallet",
            "main_category": "Fashion",
            "description": "Handcrafted minimalist wallet with RFID protection.",
            "brand": "Aura Luxe"
        },
        {
            "product_name": "Wireless Noise Cancelling Headphones",
            "main_category": "Electronics",
            "description": "High-fidelity audio with 40-hour battery life.",
            "brand": "SonicWave"
        }
    ])
    return template_df.to_csv(index=False).encode('utf-8')

def process_bulk_batch(batch_df: pd.DataFrame, model, progress_bar=None):
    """
    Processes a dataframe of products:
    1. Predicts Audience
    2. Generates Hinglish Ad Copy (Headline, Description, Slogan)
    """
    results = []
    total = len(batch_df)
    
    for i, row in batch_df.iterrows():
        p_name = str(row.get('product_name', 'Unnamed Product'))
        p_desc = str(row.get('description', ''))
        p_cat = str(row.get('main_category', 'General'))
        p_brand = str(row.get('brand', 'Premium Brand'))
        
        # 1. Predict Audience
        audience = "Professionals" # Default
        if model is not None and p_desc:
            try:
                audience = model.predict([p_desc])[0]
            except: pass
            
        # 2. Generate Copy
        copy = generate_ad_copy_template(p_name, p_cat, audience)
        
        # Compile result
        results.append({
            "Product Name": p_name,
            "Brand": p_brand,
            "Category": p_cat,
            "Predicted Audience": audience,
            "Headline": copy.get('headline', ''),
            "Ad Description": copy.get('description', ''),
            "Slogan": copy.get('slogan', '')
        })
        
        if progress_bar:
            progress_bar.progress((i + 1) / total)
            
    return pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Session State Initialisation
# ---------------------------------------------------------------------------
if "ad_content" not in st.session_state:
    st.session_state.ad_content = None
if "image_prompt" not in st.session_state:
    st.session_state.image_prompt = None
if "generated_image" not in st.session_state:
    st.session_state.generated_image = None

# ---------------------------------------------------------------------------
# Sidebar — Status Panel
# ---------------------------------------------------------------------------
with st.sidebar:
    if not HUGGINGFACE_API_KEY:
        st.warning("⚠️ HUGGINGFACE_API_KEY missing — image generation will not work.", icon="🔑")
    
    st.divider()
    st.header("⚙️ AI Settings")
    
    # Auto-detect if Cloud is better
    is_cloud = os.getenv("STREAMLIT_RUNTIME_DEBUG") is not None or llm_pipe is None
    engine_default = 0 if is_cloud else 1
    
    ai_engine = st.radio(
        "AI Prompt Engine",
        ["✨ Generate Fast (Use if loading)", "🤖 Generate Local"],
        index=engine_default,
        help="Cloud API uses a powerful 7B model. Local LLM runs on your hardware but is slower/heavier."
    )
    
    if ai_engine == "🤖 Generate Local" and llm_pipe is None:
        st.error("Local ML model not loaded/available. Using Fast Cloud fallback.")
        ai_engine = "✨ Generate Fast (Use if loading)"

# ---------------------------------------------------------------------------
# Sidebar — Product Selection
# ---------------------------------------------------------------------------
    st.header("📦 Product Selection")

    # NEW: Toggle between Catalogue, Manual, and Bulk
    app_mode = st.radio(
        "Ad Creation Mode",
        ["Catalogue", "✏️ Manual Entry", "🔍 Bulk Scanner"],
        help="Choose 'Catalogue' for pre-built products, 'Manual' for your own ideas, or 'Bulk Scanner' for high-volume uploads!"
    )
    st.divider()

    if app_mode in ["Catalogue", "✏️ Manual Entry"]:
        selected_name = ""
        product_description = ""
        category_default = "General"
        brand_default = "Aura"

        if app_mode == "Catalogue":
            if df is not None:
                # Optimization: Limit to 5,000 unique products for better selectbox speed
                all_products = df["product_name"].unique().tolist()
                product_list = sorted(all_products)[:5000] 
                
                selected_name = st.selectbox(
                    "Choose Product from Dataset",
                    product_list,
                    help="Showing top 5,000 items for faster performance.",
                )

                if "last_selected_product" not in st.session_state:
                    st.session_state.last_selected_product = selected_name

                if st.session_state.last_selected_product != selected_name:
                    st.session_state.ad_content = None
                    st.session_state.image_prompt = None
                    st.session_state.generated_image = None
                    st.session_state.last_selected_product = selected_name
                    st.rerun()

                selected_row = df[df["product_name"] == selected_name].iloc[0]
                product_description = selected_row.get("description", "")
                category_default = selected_row.get("main_category", "General")
                
                brand_default = selected_row.get("brand", "")
                if not brand_default or pd.isna(brand_default) or str(brand_default).lower() in ["unknown", "na", "nan"]:
                    words = selected_name.split() if selected_name else ["Premium"]
                    brand_default = words[0].strip(",.:'\" ")
                
                st.info(f"**Detected Category:** {category_default}")
            else:
                st.error("Dataset not found.")
                app_mode = "✏️ Manual Entry" 
        
        if app_mode == "✏️ Manual Entry":
            selected_name = st.text_input("Product Name", placeholder="e.g. Handmade Ceramic Mug")
            brand_default = st.text_input("Brand Name", placeholder="e.g. ClayArt Studio")
            product_description = st.text_area("Product Description", placeholder="Briefly describe the product...")
            
            if "last_manual_name" not in st.session_state:
                st.session_state.last_manual_name = selected_name

            if st.session_state.last_manual_name != selected_name:
                st.session_state.ad_content = None
                st.session_state.image_prompt = None
                st.session_state.generated_image = None
                st.session_state.last_manual_name = selected_name
            
            category_default = "General"

        category_list = ["Fashion", "Electronics", "Fitness", "Food", "General"]
        category_input = st.selectbox(
            "Market Category",
            category_list,
            index=category_list.index(category_default) if category_default in category_list else 4,
        )

        brand_name_input = st.text_input("Brand Name", value=brand_default)
        enable_branding = st.toggle("📸 Professional Branding Overlay", value=True)
        
        # --- ML Audience Predictor ---
        demographics_list = ["Teenagers", "Professionals", "Seniors"]
        pred_demographic = "Professionals"

        if ml_model is not None and product_description:
            try:
                pred_demographic = ml_model.predict([str(product_description)])[0]
                st.success(f"🤖 AI Predicted Audience: {pred_demographic}")
            except: pass

        demographic_input = st.selectbox(
            "Target Audience",
            demographics_list,
            index=demographics_list.index(pred_demographic) if pred_demographic in demographics_list else 1
        )

        st.divider()

        if st.button("Step 1: Compose Ad Copy ✍️", type="primary"):
            target_product = selected_name
            if not target_product:
                st.error("Please enter a product name first!")
            else:
                with st.spinner("✍️ Composing your ad copy..."):
                    st.session_state.ad_content = generate_ad_copy_template(
                        target_product, category_input, demographic_input
                    )
                with st.spinner("🤖 Crafting image prompt..."):
                    st.session_state.image_prompt = generate_image_prompt_hybrid(
                        target_product, demographic_input, st.session_state.ad_content.get("slogan", ""), 
                        brand_name_input, ai_engine, llm_pipe, HUGGINGFACE_API_KEY
                    )
                st.session_state.generated_image = None

    elif app_mode == "🔍 Bulk Scanner":
        st.info("Scanner Mode Active: Upload a file in the main area to begin.")
        st.caption("This mode ignores manual selection and processes your entire dataset.")

# ---------------------------------------------------------------------------
# Main Display
# ---------------------------------------------------------------------------
st.title("🚀 AI Product Ad Composer")

if app_mode == "🔍 Bulk Scanner":
    st.markdown("### 🔍 Bulk Ad Scanner")
    st.info("Perfect for marketing teams! Predict audience and generate Hinglish ad copy for your entire catalog in seconds.")
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        # 1. Download Template
        template_data = generate_bulk_template()
        st.download_button(
            label="📥 Download CSV Template",
            data=template_data,
            file_name="ad_composer_template.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # 2. File Upload
    uploaded_file = st.file_uploader("Step 2: Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                input_df = pd.read_csv(uploaded_file)
            else:
                input_df = pd.read_excel(uploaded_file)
            
            st.markdown(f"#### 📂 Preview: {len(input_df)} Products Found")
            st.dataframe(input_df.head(10), use_container_width=True)
            
            if st.button("🚀 Start Bulk AI Processing", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.text("🤖 AI is analyzing your products...")
                
                results_df = process_bulk_batch(input_df, ml_model, progress_bar)
                
                status_text.text("✅ All products processed successfully!")
                st.balloons()
                
                st.markdown("---")
                st.markdown("### 📊 Generated Bulk Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Download Results
                csv_results = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Final Ad Report (CSV)",
                    data=csv_results,
                    file_name="bulk_ad_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

elif st.session_state.ad_content:
    st.markdown("---")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f"""
            <div class="ad-card">
                <h3 style='color: #4F46E5;'>📝 Generated Ad Copy</h3>
                <p><strong>Headline:</strong> {st.session_state.ad_content['headline']}</p>
                <hr>
                <p><strong>Description:</strong><br>{st.session_state.ad_content['description']}</p>
                <hr>
                <p style='font-style: italic; color: #6B7280;'><strong>Slogan:</strong> {st.session_state.ad_content['slogan']}</p>
            </div>
        """, unsafe_allow_html=True)

        # --- Export & Share dropdown ---
        if st.session_state.ad_content:
            with st.expander("📝 Ad Actions (Export & Share)", expanded=True):
                # Copy Ad Copy (Text) - Always available after Step 1
                full_copy = f"Headline: {st.session_state.ad_content['headline']}\n\nDescription: {st.session_state.ad_content['description']}\n\nSlogan: {st.session_state.ad_content['slogan']}"
                st.code(full_copy, language="text")
                st.caption("☝️ Tap above to copy ad text")
                
                # Image-specific actions - Only available after Step 2
                if st.session_state.generated_image:
                    st.divider()
                    # Regenerate Button
                    if st.button("🔄 Regenerate Visual", use_container_width=True):
                        st.session_state.generated_image = None
                        with st.spinner("🎨 Generating new visual..."):
                            # Using session state to avoid scope errors
                            current_prompt = st.session_state.get('custom_prompt_input', st.session_state.image_prompt)
                            raw_image = generate_hf_image(HUGGINGFACE_API_KEY, current_prompt)
                            if raw_image:
                                pil_img = Image.open(io.BytesIO(raw_image))
                                if enable_branding:
                                    final_pil = add_branding_overlay(
                                        pil_img, brand_name_input, st.session_state.ad_content['slogan']
                                    )
                                    buf = io.BytesIO()
                                    final_pil.save(buf, format="JPEG", quality=95)
                                    st.session_state.generated_image = buf.getvalue()
                                else:
                                    st.session_state.generated_image = raw_image
                                st.rerun()
                    
                    # Download Button
                    file_name = f"{brand_name_input.replace(' ', '_')}_ad.jpg"
                    st.download_button(
                        label="📥 Download Ad Image",
                        data=st.session_state.generated_image,
                        file_name=file_name,
                        mime="image/jpeg",
                        use_container_width=True
                    )
                    
                    # Copy & Share Button (Standardized to 8501)
                    share_title = f"Ad for {selected_name}"
                    share_text = f"🚀 {st.session_state.ad_content['headline']}\n\n✨ {st.session_state.ad_content['slogan']}"
                    # Dynamic URL for sharing
                    app_url = "https://ad-visual-composer.streamlit.app" 
                    share_js = f"""
                    <script>
                    async function copyAd() {{
                        const text = `{share_text}\\n\\nCheck it out here: {app_url}`;
                        try {{
                            await navigator.clipboard.writeText(text);
                            alert('✅ Ad Copy & Link copied!');
                            if (navigator.share) {{
                                await navigator.share({{title: '{share_title}', text: text, url: '{app_url}'}});
                            }}
                        }} catch (err) {{ console.error(err); }}
                    }}
                    </script>
                    <button onclick="copyAd()" style="width: 100%; height: 3em; border-radius: 8px; background-color: #10B981; color: white; font-weight: bold; border: none; cursor: pointer; display: flex; align-items: center; justify-content: center; gap: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <span>📋 Copy & Share Ad</span>
                    </button>
                    """
                    st.components.v1.html(share_js, height=60)

    with col2:
        st.subheader("🖼️ Ad Visualization")
        custom_prompt = st.text_area(
            "Refine Visual Prompt (LLM-generated):",
            value=st.session_state.image_prompt if st.session_state.image_prompt else "Choose product and click Step 1...",
            height=140,
            key=f"prompt_input_{selected_name.replace(' ', '_')}" # Force refresh when product changes
        )

        if st.button("Step 2: Generate Visual (FLUX.1) 🚀"):
            st.session_state.generated_image = None
            if not HUGGINGFACE_API_KEY:
                st.error("🔑 Hugging Face API Key missing in .env file!")
            else:
                with st.spinner("🎨 Generating ultra-realistic visual via FLUX.1..."):
                    raw_image = generate_hf_image(
                        HUGGINGFACE_API_KEY, custom_prompt
                    )
                    
                    if raw_image:
                        # Convert bytes to PIL for processing
                        pil_img = Image.open(io.BytesIO(raw_image))
                        
                        if enable_branding:
                            with st.spinner("🖋️ Applying professional branding overlay..."):
                                final_pil = add_branding_overlay(
                                    pil_img, brand_name_input, st.session_state.ad_content['slogan']
                                )
                                # Convert back to bytes for download/sharing
                                buf = io.BytesIO()
                                final_pil.save(buf, format="JPEG", quality=95)
                                st.session_state.generated_image = buf.getvalue()
                        else:
                            st.session_state.generated_image = raw_image
                        st.rerun()

        if st.session_state.generated_image:
            st.image(
                st.session_state.generated_image,
                caption="FLUX.1 Generated High-Fidelity Ad",
                use_container_width=True,
            )
            st.success("✅ Visual generated successfully!")

else:
    st.info("👈 Use the sidebar to search for a product and start the generation process.")
    st.image(
        "https://images.unsplash.com/photo-1611162617474-5b21e879e113?q=80&w=1000&auto=format&fit=crop",
        use_container_width=True,
    )
