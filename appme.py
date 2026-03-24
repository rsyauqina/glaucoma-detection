import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from io import BytesIO
import datetime

st.set_page_config(page_title="Glaucoma Insight Dashboard", layout="centered")

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model(path="glaucoma_model.h5"):
    seq_model = tf.keras.models.load_model(path)
    inputs = tf.keras.Input(shape=(224, 224, 3))
    outputs = seq_model(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    dummy = np.zeros((1, 224, 224, 3))
    model(dummy)
    return model

model = load_model()

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
# 👁️ Mathematical Approaches to Retinal Analysis  
### Turning retinal images into meaningful insight
""")

# -------------------------------
# PATIENT INPUT
# -------------------------------
st.markdown("### 🧾 Patient Information")
patient_id = st.text_input("Enter Patient ID", placeholder="e.g. P001")

# -------------------------------
# PREPROCESS FUNCTION
# -------------------------------
def preprocess_image(image):
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0)

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Upload Retinal Image", type=["jpg","jpeg","png"])

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file is not None:

    if not patient_id:
        st.warning("⚠️ Please enter Patient ID before proceeding.")
        st.stop()

    image = Image.open(uploaded_file)
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    label = "Glaucoma" if prediction > 0.5 else "Normal"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # -------------------------------
    # SIDE-BY-SIDE LAYOUT (BALANCED)
    # -------------------------------
    col1, col2 = st.columns([1, 1])

    # LEFT: SMALLER CENTERED IMAGE
    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Retinal Image", width=250)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT: RESULT
    with col2:
        st.subheader("Prediction Result")

        if prediction > 0.7:
            st.error(f"🔴 High Risk: {label}")
        elif prediction > 0.5:
            st.warning(f"🟠 Moderate Risk: {label}")
        else:
            st.success(f"🟢 Low Risk: {label}")

        st.write(f"Confidence Score: {confidence:.4f}")

        # Human explanation
        if label == "Glaucoma":
            st.write("💬 The model detected structural patterns commonly associated with glaucoma.")
        else:
            st.write("💬 The retinal features appear within normal structural patterns.")


    # -------------------------------
    # INSIGHT SUMMARY
    # -------------------------------
    st.markdown("### 🧠 Insight Summary")

    if label == "Glaucoma":
        insight = (
            "This result suggests a higher likelihood of glaucoma. "
            "The model identified patterns that differ from typical healthy retinal images. "
            "While this is not a medical diagnosis, it highlights the importance of early clinical evaluation."
        )
    else:
        insight = (
            "This result suggests that the retinal structure appears normal. "
            "No strong indicators of glaucoma were detected. "
            "Regular eye check-ups are still recommended for long-term monitoring."
        )

    st.info(insight)

    # -------------------------------
    # PDF REPORT
    # -------------------------------
    def create_pdf(image, label, confidence, insight, patient_id):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter

        c.setFont("Helvetica-Bold", 18)
        c.drawCentredString(width/2, height-50, "AI-Assisted Retinal Insight Report")

        c.setFont("Helvetica", 11)
        date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.drawString(50, height-80, f"Date: {date_str}")

        c.drawString(50, height-100, f"Patient ID: {patient_id}")
        c.drawString(50, height-120, f"Predicted Class: {label}")
        c.drawString(50, height-140, f"Confidence Score: {confidence:.4f}")

        text = c.beginText(50, height-180)
        text.setFont("Helvetica", 11)
        for line in insight.split(". "):
            text.textLine(line.strip())
        c.drawText(text)

        c.drawImage(ImageReader(image), 50, height-450, width=200, height=200)

        c.showPage()
        c.save()
        buffer.seek(0)
        return buffer

    pdf_buffer = create_pdf(image, label, confidence, insight, patient_id)

    st.download_button(
        "📄 Download Patient Report",
        data=pdf_buffer,
        file_name=f"{patient_id}_retinal_report.pdf",
        mime="application/pdf"
    )

# -------------------------------
# RESPONSIBLE AI
# -------------------------------
st.markdown("### 📊 Responsible Use of Data")
st.write(
    "This system is designed for awareness and early screening only. "
    "It does not replace professional medical diagnosis. "
    "Please consult an eye specialist for clinical decisions."
)