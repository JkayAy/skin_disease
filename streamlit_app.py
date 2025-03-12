import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import logging
import base64
import cv2
import tensorflow as tf
from anthropic import Anthropic
from openai import OpenAI
from dotenv import load_dotenv
import json
from datetime import datetime

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# App Configuration & Setup
# ==========================
st.set_page_config(
    page_title="AI Dermatology Diagnostic Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🩺"
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to bottom, #333, #111);
        color: white;
    }
    .sidebar .sidebar-content {
        background: #333;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🩺 AI Dermatology Diagnostic Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #21abcd; font-size: 1.2em;'>Upload a skin image for AI-powered analysis</p>", unsafe_allow_html=True)

# ==========================
# Global Variables
# ==========================
# Define skin disease classes (adjust as needed)
# SKIN_CLASSES = [
#     'Melanoma', 
#     'Basal Cell Carcinoma', 
#     'Squamous Cell Carcinoma', 
#     'Benign Nevus', 
#     'Seborrheic Keratosis', 
#     'Dermatofibroma', 
#     'Vascular Lesion'
# ]

# Define skin disease classes (now updated to match 14 output values)
SKIN_CLASSES = [
    'Melanoma', 
    'Basal Cell Carcinoma', 
    'Squamous Cell Carcinoma', 
    'Benign Nevus', 
    'Seborrheic Keratosis', 
    'Dermatofibroma', 
    'Vascular Lesion',
    'Actinic Keratosis',
    'Lichen Planus',
    'Psoriasis',
    'Atopic Dermatitis',
    'Impetigo',
    'Rosacea',
    'Eczema'
]


# ==========================
# Patient Record Management
# ==========================
class PatientRecord:
    def __init__(self):
        self.medical_history = {}
        self.current_symptoms = {}
        self.previous_diagnoses = []
        self.medications = []
        self.allergies = []
    
    def add_medical_history(self, condition, details):
        self.medical_history[condition] = details
    
    def add_current_symptoms(self, symptom, severity):
        self.current_symptoms[symptom] = severity
    
    def add_previous_diagnosis(self, diagnosis, date):
        self.previous_diagnoses.append({"diagnosis": diagnosis, "date": str(date)})
    
    def add_medication(self, medication, dosage):
        self.medications.append({"medication": medication, "dosage": dosage})
    
    def add_allergy(self, allergen):
        self.allergies.append(allergen)
    
    def get_comprehensive_history(self):
        return {
            "Medical History": self.medical_history,
            "Current Symptoms": self.current_symptoms,
            "Previous Diagnoses": self.previous_diagnoses,
            "Medications": self.medications,
            "Allergies": self.allergies
        }

def collect_patient_information() -> PatientRecord:
    st.sidebar.header("🩺 Patient Information")
    patient_record = PatientRecord()
    with st.sidebar.form(key="patient_info_form"):
        st.subheader("Medical History")
        history_condition = st.text_input("Condition")
        history_details = st.text_area("Details")
        
        st.subheader("Current Symptoms")
        symptom = st.text_input("Symptom")
        severity = st.select_slider("Severity", options=['Mild', 'Moderate', 'Severe'])
        
        st.subheader("Previous Diagnoses")
        diagnosis = st.text_input("Diagnosis")
        diagnosis_date = st.date_input("Date")
        
        st.subheader("Medications")
        medication = st.text_input("Medication")
        dosage = st.text_input("Dosage")
        
        st.subheader("Allergies")
        allergen = st.text_input("Allergen")
        
        submitted = st.form_submit_button("Save Patient Info")
        if submitted:
            if history_condition.strip() and history_details.strip():
                patient_record.add_medical_history(history_condition, history_details)
            if symptom.strip():
                patient_record.add_current_symptoms(symptom, severity)
            if diagnosis.strip() and diagnosis_date:
                patient_record.add_previous_diagnosis(diagnosis, diagnosis_date)
            if medication.strip() and dosage.strip():
                patient_record.add_medication(medication, dosage)
            if allergen.strip():
                patient_record.add_allergy(allergen)
    # Display the saved patient information summary below the form
    # st.sidebar.markdown("### 📋 Patient Information Summary")
    # st.sidebar.json(patient_record.get_comprehensive_history())
    return patient_record

# ==========================
# GradCAM Functions
# ==========================
def compute_gradcam(model, preprocessed_image, class_idx, layer_name='conv5_block16_2_conv'):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(preprocessed_image)
        class_score = predictions[:, class_idx]
    grads = tape.gradient(class_score, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def apply_gradcam_overlay(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

# ==========================
# Model and API Client Loading Functions
# ==========================
# @st.cache(allow_output_mutation=True)
@st.cache_resource
def load_dermatology_model():
    try:
        # Load your pre-trained dermatology model from my_model.keras
        model = load_model("my_model.keras")
        # Do not call st.write() inside a cached function
        logger.info("Dermatology model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error loading dermatology model: {str(e)}")
        raise

# @st.cache(allow_output_mutation=True)
@st.cache_resource
def load_feature_extractor():
    try:
        model = ResNet50(weights='imagenet', include_top=True)
        logger.info("Successfully loaded ResNet50 feature extractor")
        return model
    except Exception as e:
        logger.error(f"Error loading ResNet50 feature extractor: {str(e)}")
        raise

# @st.cache(allow_output_mutation=True)
@st.cache_resource
def load_gpt4_vision_client():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        client = OpenAI(api_key=api_key)
        logger.info("Successfully initialized GPT-4 Vision client")
        return client
    except Exception as e:
        logger.error(f"Error initializing GPT-4 Vision client: {str(e)}")
        raise

# @st.cache(allow_output_mutation=True)
@st.cache_resource
def load_claude_client():
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not found in environment variables")
        client = Anthropic(api_key=api_key)
        logger.info("Successfully initialized Claude 3 client")
        return client
    except Exception as e:
        logger.error(f"Error initializing Claude 3 client: {str(e)}")
        raise

# ==========================
# Image Processing Functions
# ==========================
def preprocess_image(image, target_size=(320, 320)):
    try:
        image = image.convert("RGB")
        image = image.resize(target_size)
        image = np.array(image) / 255.0
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def extract_image_features(image: Image.Image) -> str:
    try:
        model = load_feature_extractor()
        img = image.resize((224, 224))
        x = np.array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        features = decode_predictions(preds, top=3)[0]
        descriptions = [f"{label} ({score:.2%})" for _, label, score in features]
        return "Image appears to show: " + ", ".join(descriptions)
    except Exception as e:
        logger.error(f"Error extracting image features: {str(e)}")
        return "Unable to extract image features"

# ==========================
# Analysis Functions for LLMs
# ==========================
def get_dermatology_analysis_prompt(image_description: str, patient_history: dict) -> str:
    if not any(patient_history.values()):
        return f"""As an expert dermatologist, please analyze the following skin image:

Image Features: {image_description}

Provide a structured analysis including:
1. Initial observations (e.g., texture, color, irregularities)
2. Potential diagnoses
3. Recommendations for further tests or consultations

IMPORTANT: This analysis is AI-assisted and should be verified by a qualified healthcare professional."""
    else:
        history_str = json.dumps(patient_history, indent=2)
        return f"""As an expert dermatologist, please analyze the following skin image considering the patient’s history:

Patient History:
{history_str}

Image Features: {image_description}

Provide a structured analysis including:
1. Initial observations (e.g., skin texture, color variations, border irregularities)
2. Potential diagnoses (with differential considerations)
3. Recommendations for further tests or consultations

IMPORTANT: This analysis is AI-assisted and must be reviewed by a qualified healthcare professional."""

def analyze_with_gpt4(image_path: str, patient_record: PatientRecord) -> str:
    try:
        client = load_gpt4_vision_client()
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
        # Use extracted image features as a description (could be replaced with a dedicated feature extractor)
        image_description = extract_image_features(Image.open(image_path))
        prompt = get_dermatology_analysis_prompt(image_description, patient_record.get_comprehensive_history())
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                ]
            }],
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"GPT-4 Vision analysis failed: {str(e)}")
        raise

def analyze_with_claude(image_path: str, patient_record: PatientRecord) -> str:
    try:
        client = load_claude_client()
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
        image_description = extract_image_features(Image.open(image_path))
        prompt = get_dermatology_analysis_prompt(image_description, patient_record.get_comprehensive_history())
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}}
                ]
            }]
        )
        return response.content[0].text
    except Exception as e:
        logger.error(f"Claude 3 analysis failed: {str(e)}")
        raise

def analyze_with_medpalm(image_path: str, patient_record: PatientRecord) -> str:
    image_description = extract_image_features(Image.open(image_path))
    prompt = get_dermatology_analysis_prompt(image_description, patient_record.get_comprehensive_history())
    # Placeholder analysis for MedPaLM
    analysis = (
        "MedPaLM Analysis Result:\n\n"
        "Initial Observations:\n"
        "The image shows a skin lesion with irregular borders and varying color intensity.\n\n"
        "Potential Diagnoses:\n"
        "Possible melanoma or other skin conditions; further tests are recommended.\n\n"
        "Recommendations:\n"
        "A biopsy and consultation with a dermatologist are advised.\n\n"
        "IMPORTANT: This analysis is AI-assisted and must be reviewed by a qualified healthcare professional."
    )
    return analysis

def interpret_for_child(text: str) -> str:
    simplified = (
        "Here's a simple explanation:\n"
        "The analysis shows that the skin area has some unusual patterns, which might mean there is a condition that needs a doctor's attention. "
        "Doctors would perform further tests to be sure. This is a simplified explanation."
    )
    return simplified

# ==========================
# Sidebar UI Component
# ==========================
def create_sidebar() -> PatientRecord:
    patient_record = collect_patient_information()
    st.sidebar.markdown("### 📋 Patient Information Summary")
    st.sidebar.json(patient_record.get_comprehensive_history())
    return patient_record

# ==========================
# Main Application UI
# ==========================
def main():
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'diagnosis_result' not in st.session_state:
        st.session_state.diagnosis_result = {"gpt4": None, "claude": None, "medpalm": None}

    # Create top menu tabs for analysis modes
    tabs = st.tabs(["Deep Learning Model", "GPT-4 Vision", "Claude 3", "MedPaLM Analysis"])

    # File uploader placed below the menu
    st.markdown("### Upload Skin Image")
    uploaded_file = st.file_uploader("Upload a skin image (png, jpg, jpeg)", type=["png", "jpg", "jpeg"], key="main_uploader")
    st.session_state.uploaded_file = uploaded_file

    # Display the uploaded image preview (if available)
    if st.session_state.uploaded_file is not None:
        st.image(st.session_state.uploaded_file, caption="Uploaded Image", width=300, use_column_width=True)

    # Collect patient information from the sidebar
    patient_record = create_sidebar()

    if st.session_state.uploaded_file is not None:
        image_path = None
        try:
            # Create temporary file for uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(st.session_state.uploaded_file.getvalue())
                image_path = tmp_file.name

            # -------------------------
            # Deep Learning Model Tab
            # -------------------------
            # -------------------------
# Deep Learning Model Tab
# -------------------------
            with tabs[0]:
                st.markdown("### Deep Learning Model Analysis")
                try:
                    model = load_dermatology_model()
                    processed_image = preprocess_image(Image.open(st.session_state.uploaded_file))
                    predictions = model.predict(processed_image)[0]
                    
                    # Debug: Display predictions shape and values
                    st.write("Predictions:", predictions, "Shape:", predictions.shape)
                    
                    if len(predictions) != len(SKIN_CLASSES):
                        st.error(f"Mismatch between predicted outputs ({len(predictions)}) and defined skin classes ({len(SKIN_CLASSES)}).")
                    else:
                        predicted_idx = int(np.argmax(predictions))
                        predicted_class = SKIN_CLASSES[predicted_idx]
                        confidence = predictions[predicted_idx]
                        st.success(f"Deep Learning Prediction: **{predicted_class}** (Confidence: {confidence:.2f})")
                        
                        # GradCAM Visualization
                        try:
                            original_img = cv2.cvtColor(np.array(Image.open(st.session_state.uploaded_file).convert("RGB")), cv2.COLOR_RGB2BGR)
                            heatmap = compute_gradcam(model, processed_image, predicted_idx)
                            overlay_img = apply_gradcam_overlay(original_img, heatmap)
                            overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
                            st.markdown("### GradCAM Visualization")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.image(original_img, caption="Original Image", use_column_width=True)
                            with col2:
                                st.image(overlay_img, caption="GradCAM Overlay", use_column_width=True)
                        except Exception as e:
                            st.error(f"GradCAM visualization failed: {str(e)}")
                except Exception as e:
                    st.error(f"Deep learning analysis failed: {str(e)}")


            # with tabs[0]:
            #     st.markdown("### Deep Learning Model Analysis")
            #     try:
            #         model = load_dermatology_model()
            #         processed_image = preprocess_image(Image.open(st.session_state.uploaded_file))
            #         predictions = model.predict(processed_image)[0]
            #         predicted_idx = int(np.argmax(predictions))
            #         predicted_class = SKIN_CLASSES[predicted_idx]
            #         confidence = predictions[predicted_idx]
                    
            #         st.success(f"Deep Learning Prediction: **{predicted_class}** (Confidence: {confidence:.2f})")
                    
            #         # GradCAM Visualization
            #         try:
            #             original_img = cv2.cvtColor(np.array(Image.open(st.session_state.uploaded_file).convert("RGB")), cv2.COLOR_RGB2BGR)
            #             heatmap = compute_gradcam(model, processed_image, predicted_idx)
            #             overlay_img = apply_gradcam_overlay(original_img, heatmap)
            #             overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
            #             st.markdown("### GradCAM Visualization")
            #             col1, col2 = st.columns(2)
            #             with col1:
            #                 st.image(original_img, caption="Original Image", use_column_width=True)
            #             with col2:
            #                 st.image(overlay_img, caption="GradCAM Overlay", use_column_width=True)
            #         except Exception as e:
            #             st.error(f"GradCAM visualization failed: {str(e)}")
            #     except Exception as e:
            #         st.error(f"Deep learning analysis failed: {str(e)}")

            # -------------------------
            # GPT-4 Vision Analysis Tab
            # -------------------------
            with tabs[1]:
                st.markdown("### GPT-4 Vision Analysis with Patient Context")
                if st.button("Run Diagnosis (GPT-4)"):
                    with st.spinner("Generating GPT-4 analysis..."):
                        try:
                            st.session_state.diagnosis_result["gpt4"] = analyze_with_gpt4(image_path, patient_record)
                        except Exception as e:
                            st.error(f"GPT-4 analysis failed: {str(e)}")
                if st.session_state.diagnosis_result["gpt4"]:
                    st.markdown(
                        f"""<div style='background-color: #1A2833; padding: 1.5rem;
                        border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                            {st.session_state.diagnosis_result["gpt4"]}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    if st.button("Explain GPT-4 Diagnosis for Child"):
                        simplified = interpret_for_child(st.session_state.diagnosis_result["gpt4"])
                        st.markdown(
                            f"""<div style='background-color: #1A2833; padding: 1.5rem;
                            border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                                {simplified}
                            </div>""", unsafe_allow_html=True
                        )

            # -------------------------
            # Claude 3 Analysis Tab
            # -------------------------
            with tabs[2]:
                st.markdown("### Claude 3 Analysis with Patient Context")
                if st.button("Run Diagnosis (Claude 3)"):
                    with st.spinner("Generating Claude 3 analysis..."):
                        try:
                            st.session_state.diagnosis_result["claude"] = analyze_with_claude(image_path, patient_record)
                        except Exception as e:
                            st.error(f"Claude 3 analysis failed: {str(e)}")
                if st.session_state.diagnosis_result["claude"]:
                    st.markdown(
                        f"""<div style='background-color: #1A2833; padding: 1.5rem;
                        border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                            {st.session_state.diagnosis_result["claude"]}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    if st.button("Explain Claude 3 Diagnosis for Child"):
                        simplified = interpret_for_child(st.session_state.diagnosis_result["claude"])
                        st.markdown(
                            f"""<div style='background-color: #1A2833; padding: 1.5rem;
                            border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                                {simplified}
                            </div>""", unsafe_allow_html=True
                        )

            # -------------------------
            # MedPaLM Analysis Tab
            # -------------------------
            with tabs[3]:
                st.markdown("### MedPaLM Analysis with Patient Context")
                if st.button("Run Diagnosis (MedPaLM)"):
                    with st.spinner("Generating MedPaLM analysis..."):
                        try:
                            st.session_state.diagnosis_result["medpalm"] = analyze_with_medpalm(image_path, patient_record)
                        except Exception as e:
                            st.error(f"MedPaLM analysis failed: {str(e)}")
                if st.session_state.diagnosis_result["medpalm"]:
                    st.markdown(
                        f"""<div style='background-color: #1A2833; padding: 1.5rem;
                        border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                            {st.session_state.diagnosis_result["medpalm"]}
                        </div>""",
                        unsafe_allow_html=True
                    )
                    if st.button("Explain MedPaLM Diagnosis for Child"):
                        simplified = interpret_for_child(st.session_state.diagnosis_result["medpalm"])
                        st.markdown(
                            f"""<div style='background-color: #1A2833; padding: 1.5rem;
                            border-radius: 10px; border-left: 4px solid #2E86C1; color: white;'>
                                {simplified}
                            </div>""", unsafe_allow_html=True
                        )

        except Exception as e:
            st.error(f"File processing error: {str(e)}")
        finally:
            if image_path and os.path.exists(image_path):
                os.unlink(image_path)
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center;'>🏥 Powered by Advanced AI for Medical Professionals<br>"
        "<small>For assistance purposes only. Consult a healthcare provider for medical decisions.</small></p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("An unexpected error occurred. Please try again.")
        logger.error(f"Application error: {str(e)}")
