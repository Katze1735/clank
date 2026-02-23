import streamlit as st
import numpy as np
import requests
from PIL import Image, ImageOps
from io import BytesIO
from tensorflow.keras.models import load_model
from supabase import create_client, Client

# -----------------------------
# CONFIG
# -----------------------------
SUPABASE_URL = "YOUR_SUPABASE_URL"
SUPABASE_KEY = "YOUR_SUPABASE_ANON_KEY"
TABLE_NAME = "images"

# -----------------------------
# LOAD MODEL (cached)
# -----------------------------
@st.cache_resource
def load_tf_model():
    model = load_model("keras_model.h5", compile=False)
    class_names = open("labels.txt", "r").readlines()
    return model, class_names

model, class_names = load_tf_model()

# -----------------------------
# CONNECT SUPABASE
# -----------------------------
@st.cache_resource
def init_supabase():
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase: Client = init_supabase()

# -----------------------------
# IMAGE PREPROCESSING
# -----------------------------
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# -----------------------------
# PREDICT FUNCTION
# -----------------------------
def predict_image(image):
    data = preprocess_image(image)
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence_score = float(prediction[0][index])
    return class_name, confidence_score

# -----------------------------
# FETCH IMAGES FROM SUPABASE
# -----------------------------
@st.cache_data(ttl=60)
def fetch_images():
    response = supabase.table(TABLE_NAME).select("*").execute()
    return response.data

# -----------------------------
# UI
# -----------------------------
st.title("🧠 AI Image Classifier Dashboard")

images = fetch_images()

if not images:
    st.warning("No images found in database.")
    st.stop()

# Predict for all images
results = []

with st.spinner("Running predictions..."):
    for entry in images:
        try:
            img_response = requests.get(entry["image_url"])
            img = Image.open(BytesIO(img_response.content)).convert("RGB")
            class_name, confidence = predict_image(img)

            results.append({
                "id": entry["id"],
                "image_url": entry["image_url"],
                "class_name": class_name,
                "confidence": confidence
            })

        except Exception as e:
            st.error(f"Error processing image {entry['id']}: {e}")

# -----------------------------
# FILTERING
# -----------------------------
all_classes = sorted(list(set(r["class_name"] for r in results)))

selected_class = st.selectbox("Filter by Category", ["All"] + all_classes)
confidence_threshold = st.slider("Minimum Confidence", 0.0, 1.0, 0.5)

filtered_results = [
    r for r in results
    if (selected_class == "All" or r["class_name"] == selected_class)
    and r["confidence"] >= confidence_threshold
]

st.write(f"Showing {len(filtered_results)} results")

# -----------------------------
# DISPLAY IMAGES
# -----------------------------
cols = st.columns(3)

for i, result in enumerate(filtered_results):
    with cols[i % 3]:
        st.image(result["image_url"])
        st.write(f"**Class:** {result['class_name']}")
        st.write(f"Confidence: {result['confidence']:.2f}")
