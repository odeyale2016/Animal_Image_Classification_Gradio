##############################################################
# Animal Image Classification using ResNet18
# Developed by: Odeyale Kehinde Musiliudeen
##############################################################

import os
import time
from datetime import datetime

import torch
import torch.nn as nn

import pandas as pd

import plotly.express as px

import gradio as gr

from PIL import Image

from torchvision import models, transforms

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Table,
    TableStyle,
    Spacer
)

##############################################################
# Configuration
##############################################################

CLASS_NAMES = [
    "Butterfly",
    "Cat",
    "Chicken",
    "Cow",
    "Dog",
    "Elephant",
    "Horse",
    "Sheep",
    "Spider",
    "Squirrel"
]

MODEL_PATH = "animal_classifier_resnet18.pth"

CONFIDENCE_THRESHOLD = 0.70

MODEL_NAME = "ResNet18"

MODEL_SIZE = "42 MB"

##############################################################
# Create folders automatically
##############################################################

os.makedirs("reports", exist_ok=True)

##############################################################
# Load Model
##############################################################

@torch.no_grad()
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(CLASS_NAMES)
    )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(state)

    model.eval()

    return model


model = load_model()

##############################################################
# Image Preprocessing
##############################################################

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485,0.456,0.406],

        std=[0.229,0.224,0.225]

    )

])

##############################################################
# CSV Report
##############################################################

def create_csv(top5_df):

    filename = os.path.join(

        "reports",

        f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    )

    top5_df.to_csv(filename,index=False)

    return filename

##############################################################
# PDF Report
##############################################################

def create_pdf(

    prediction,

    confidence,

    inference_time,

    top5_df

):

    filename = os.path.join(

        "reports",

        f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    )

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(

        Paragraph(

            "Animal Image Classification Report",

            styles["Title"]

        )

    )

    elements.append(Spacer(1,15))

    elements.append(

        Paragraph(

            f"<b>Date:</b> {datetime.now()}",

            styles["Normal"]

        )

    )

    elements.append(

        Paragraph(

            f"<b>Prediction:</b> {prediction}",

            styles["Normal"]

        )

    )

    elements.append(

        Paragraph(

            f"<b>Confidence:</b> {confidence}",

            styles["Normal"]

        )

    )

    elements.append(

        Paragraph(

            f"<b>Inference Time:</b> {inference_time}",

            styles["Normal"]

        )

    )

    elements.append(Spacer(1,15))

    table_data = [list(top5_df.columns)]

    table_data.extend(

        top5_df.values.tolist()

    )

    table = Table(table_data)

    table.setStyle(

        TableStyle([

            ("BACKGROUND",(0,0),(-1,0),colors.darkgreen),

            ("TEXTCOLOR",(0,0),(-1,0),colors.white),

            ("GRID",(0,0),(-1,-1),1,colors.black),

            ("BACKGROUND",(0,1),(-1,-1),colors.beige),

            ("ALIGN",(0,0),(-1,-1),"CENTER")

        ])

    )

    elements.append(table)

    doc.build(elements)

    return filename

##############################################################
# Prediction Function
##############################################################

def predict(image):

    if image is None:

        return (

            None,

            "Please upload an image.",

            "",

            "",

            "",

            pd.DataFrame(),

            None,

            None,

            None

        )

    start = time.time()

    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(tensor)

        probabilities = torch.softmax(

            output,

            dim=1

        )[0]

    inference_time = round(

        time.time()-start,

        3

    )

    confidence,pred = torch.max(

        probabilities,

        0

    )

    confidence = confidence.item()

    predicted_class = CLASS_NAMES[pred.item()]

    if confidence >= 0.90:

        status = "🟢 High Confidence"

    elif confidence >= 0.75:

        status = "🟡 Medium Confidence"

    else:

        status = "🔴 Low Confidence"

    if confidence < CONFIDENCE_THRESHOLD:

        prediction_html = f"""

        <h2 style='color:red;'>

        Not confident enough

        </h2>

        <p>

        Best Guess:

        <b>{predicted_class}</b>

        </p>

        """

    else:

        prediction_html = f"""

        <h1 style='color:green;'>

        {predicted_class}

        </h1>

        """

    ##########################################################
    # Top-5 Predictions
    ##########################################################

    top5 = torch.topk(

        probabilities,

        5

    )

    top5_df = pd.DataFrame({

        "Rank":

        range(1,6),

        "Animal":[

            CLASS_NAMES[i]

            for i in top5.indices.tolist()

        ],

        "Probability (%)":[

            round(

                float(v)*100,

                2

            )

            for v in top5.values.tolist()

        ]

    })

    ##########################################################
    # Plotly Confidence Chart
    ##########################################################

    fig = px.bar(

        top5_df,

        x="Probability (%)",

        y="Animal",

        orientation="h",

        text="Probability (%)",

        color="Probability (%)",

        color_continuous_scale="Viridis"

    )

    fig.update_layout(

        height=420,

        template="plotly_white",

        coloraxis_showscale=False,

        title="Top-5 Prediction Confidence"

    )

    fig.update_traces(

        texttemplate="%{text:.2f}%",

        textposition="outside"

    )

    ##########################################################
    # Model Information
    ##########################################################

    model_information = f"""

Architecture : {MODEL_NAME}

Framework : PyTorch

Classes : {len(CLASS_NAMES)}

Input Size : 224 × 224

Model Size : {MODEL_SIZE}

Confidence Threshold : {CONFIDENCE_THRESHOLD}

Prediction Status : {status}

"""

    ##########################################################
    # Reports
    ##########################################################

    csv_file = create_csv(top5_df)

    pdf_file = create_pdf(

        predicted_class,

        f"{confidence*100:.2f}%",

        f"{inference_time:.3f} sec",

        top5_df

    )

    ##########################################################
    # Return Outputs
    ##########################################################

    return (

        image,

        prediction_html,

        f"{confidence*100:.2f}%",

        f"{inference_time:.3f} sec",

        model_information,

        top5_df,

        fig,

        csv_file,

        pdf_file

    )


##############################################################
# User Interface
##############################################################

with gr.Blocks(
    title="Animal Image Classification"
) as demo:

    gr.Markdown(
        """
# 🐾 Animal Image Classification using Deep Learning

This application uses a **ResNet18** convolutional neural network trained to classify **10 animal species**.

### Supported Classes

Butterfly • Cat • Chicken • Cow • Dog • Elephant • Horse • Sheep • Spider • Squirrel
"""
    )

    ##########################################################
    # Main Layout
    ##########################################################

    with gr.Row():

        ######################################################
        # Left Panel
        ######################################################

        with gr.Column(scale=1):

            input_image = gr.Image(
                type="pil",
                label="Upload Animal Image"
            )

            with gr.Row():

                predict_btn = gr.Button(
                    "🔍 Predict",
                    variant="primary"
                )

                clear_btn = gr.ClearButton(
                    components=[input_image]
                )

        ######################################################
        # Right Panel
        ######################################################

        with gr.Column(scale=1):

            output_image = gr.Image(
                label="Uploaded Image"
            )

            prediction = gr.HTML(
                label="Prediction"
            )

            confidence = gr.Textbox(
                label="Confidence"
            )

            inference_time = gr.Textbox(
                label="Inference Time"
            )

    ##########################################################
    # Top-5 Prediction + Confidence Chart
    ##########################################################

    gr.Markdown("## 📊 Prediction Analysis")

    with gr.Row():

        with gr.Column():

            top5_table = gr.Dataframe(
                headers=[
                    "Rank",
                    "Animal",
                    "Probability (%)"
                ],
                interactive=False,
                label="Top-5 Predictions"
            )

        with gr.Column():

            confidence_chart = gr.Plot(
                label="Confidence Distribution"
            )

    ##########################################################
    # Model Information
    ##########################################################

    with gr.Accordion(
        "ℹ️ Model Information",
        open=False
    ):

        info = gr.Textbox(
            lines=8,
            interactive=False,
            label="Model Details"
        )

    ##########################################################
    # Download Reports
    ##########################################################

    gr.Markdown("## 📥 Download Reports")

    with gr.Row():

        csv_download = gr.File(
            label="CSV Report"
        )

        pdf_download = gr.File(
            label="PDF Report"
        )

    ##########################################################
    # Footer
    ##########################################################

    gr.Markdown(
        """
---

### Developed by

**Odeyale Kehinde Musiliudeen**

Alphatech AI Research Laboratory

Powered by **PyTorch + Gradio**
"""
    )

    ##########################################################
    # Prediction Event
    ##########################################################

    predict_btn.click(

        fn=predict,

        inputs=input_image,

        outputs=[

            output_image,

            prediction,

            confidence,

            inference_time,

            info,

            top5_table,

            confidence_chart,

            csv_download,

            pdf_download

        ]

    )

##############################################################
# Launch Application
##############################################################

import os

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        debug=False,
        show_error=True
    )