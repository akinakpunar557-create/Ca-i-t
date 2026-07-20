import json
import time
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ---------------------- CONFIG ----------------------
st.set_page_config(
    page_title="🐱 Cat Breed Classifier",
    page_icon="🐱",
    layout="wide"
)

MODEL_PATH = "efficientnet_b0_feature_extractor.pth"
CLASSES_PATH = "class_names.json"
EXAMPLE_IMAGE = "ragdoll.jpg"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------- LOAD ----------------------
@st.cache_resource
def load_classes():
    with open(CLASSES_PATH) as f:
        return json.load(f)

@st.cache_resource
def load_model(num_classes):
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(image, model, class_names, top_k=5):
    x = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)
        p, idx = probs.topk(top_k)

    return [(class_names[i], float(prob))
            for i, prob in zip(idx[0], p[0])]

# ---------------------- DATA ----------------------
class_names = load_classes()
model = load_model(len(class_names))

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.title("🐱 Cat Classifier")
    st.markdown("### Model")
    st.write("EfficientNet-B0")
    st.write(f"Classes: **{len(class_names)}**")
    st.write(f"Device: **{device}**")
    st.markdown("---")
    st.info(
        "Upload a clear image of a cat. "
        "The model predicts the most likely breed."
    )

# ---------------------- HEADER ----------------------
st.title("🐱 Cat Breed Classifier")
st.write("Upload a cat image or try the built-in example.")

col_upload, col_button = st.columns([3,1])

with col_upload:
    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg","jpeg","png"]
    )

with col_button:
    use_example = st.button("🖼 Example")

image = None
caption = ""

if uploaded:
    image = Image.open(uploaded)
    caption = "Uploaded Image"

elif use_example:
    image = Image.open(EXAMPLE_IMAGE)
    caption = "Example Image"

# ---------------------- PREDICTION ----------------------
if image:

    left, right = st.columns([1.2,1])

    with left:
        st.image(image, caption=caption, use_container_width=True)

    with right:

        start = time.time()

        with st.spinner("🔍 Predicting..."):
            results = predict(image, model, class_names)

        elapsed = (time.time()-start)*1000

        best, conf = results[0]

        st.success(f"## 🏆 {best}")

        st.metric(
            "Confidence",
            f"{conf*100:.2f}%"
        )

        st.metric(
            "Inference Time",
            f"{elapsed:.1f} ms"
        )

        if conf < 0.50:
            st.warning(
                "Low confidence prediction. "
                "Try a clearer image."
            )

        st.markdown("### Top Predictions")

        for breed, prob in results:
            st.write(f"**{breed}**")
            st.progress(float(prob))
            st.caption(f"{prob*100:.2f}%")

else:
    st.info("⬆ Upload an image to begin.")

# ---------------------- FOOTER ----------------------
st.markdown("---")

with st.expander("About"):
    st.write("""
This application classifies cat breeds using a fine-tuned
EfficientNet-B0 convolutional neural network.

**Model**
- EfficientNet-B0
- Transfer Learning
- PyTorch
- Streamlit

Developed for educational and portfolio purposes.
""")