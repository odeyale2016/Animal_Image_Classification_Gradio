import streamlit as st
import torch  
import torch.nn as nn  
from torchvision import models, transforms  
from PIL import Image


# --- Config ---
CLASS_NAMES = ['Butterfly', 'Cat', 'Chicken','Cow', 'Dog', 'Elephant', 'Horse', 'Sheep', 'Spider','Squirrel']  # replace with your actual class_names, in the same order
MODEL_PATH = 'animal_classifier_resnet18.pth'
CONFIDENCE_THRESHOLD = 0.70 
# --- Load model (cached so it only loads once, not on every interaction) ---
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)  # architecture only, no pretrained weights needed here
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# --- Same transform used for validation/inference during training ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# --- UI ---
st.title("Animal Image Classifier ")

st.write("This app uses a pre-trained ResNet18 model to classify images of animals. ")
st.write("The model was trained on a dataset of 10 different animals, and it can predict which animal is in the uploaded image.")   

st.write("Upload an image and the model will predict which animal it is.")

st.caption(
    f"⚠️ This model can only recognize: **{', '.join(CLASS_NAMES)}**. "
    f"Uploading other animals or objects may produce incorrect or misleading results."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width='stretch')

    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()

    if confidence < CONFIDENCE_THRESHOLD:
        st.subheader("Prediction: Not confident enough to classify")
        st.write(
            f"The model's best guess was **{CLASS_NAMES[predicted_idx]}** "
            f"at only {confidence * 100:.2f}% confidence, which is below the "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}% threshold. This usually means the "
            f"uploaded image isn't one of the animals this model was trained on."
        )
    else:
        st.subheader(f"Prediction: **{CLASS_NAMES[predicted_idx]}**")
        st.write(f"Confidence: {confidence * 100:.2f}%")

    # Show all class probabilities regardless, for transparency
    st.write("All class probabilities:")
    for i, class_name in enumerate(CLASS_NAMES):
        st.write(f"- {class_name}: {probabilities[i].item() * 100:.2f}%")

    st.write("Developed by: Odeyale Kehinde Musiliudeen(Alphatech)")