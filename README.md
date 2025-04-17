

# A distilBERT based Phishing Email Detection Model

## Model Overview
This model is based on DistilBERT and has been fine-tuned for multilabel classification of Emails and URLs as safe or potentially phishing.

## Key Specifications
- __Base Architecture:__ DistilBERT
- __Task:__ Multilabel Classification
- __Fine-tuning Framework:__ Hugging Face Trainer API
- __Training Duration:__ 3 epochs

## Performance Metrics
- __Accuracy:__ 99.58
- __F1-score:__ 99.579
- __Precision:__ 99.583
- __Recall:__ 99.58

## Dataset Details

The model was trained on a custom dataset of Emails and URLs labeled as legitimate or phishing. The dataset is available at [`cybersectony/PhishingEmailDetectionv2.0`](https://huggingface.co/datasets/cybersectony/PhishingEmailDetectionv2.0) on the Hugging Face Hub.


## Usage Guide

## Installation

```bash
pip install transformers
pip install torch
pip install gradio
```

## Possible Issue

If you receive a UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy then run the following:

```python
pip install "numpy>=1.16.5,<1.23.0"
```
## Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#tokenizer = AutoTokenizer.from_pretrained("cybersectony/phishing-email-detection-distilbert_v2.4.1")
import torch
import gradio as gr
import webbrowser

# Load model and tokenizer
model_name = "cybersectony/phishing-email-detection-distilbert_v2.4.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define the prediction function
def predict_email(email_text):
    # Preprocess and tokenize
    inputs = tokenizer(
        email_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get probabilities for each class
    probs = predictions[0].tolist()

    # Create labels dictionary
    labels = {
        "legitimate_email": probs[0],
        "phishing_url": probs[1],
        "legitimate_url": probs[2],
        "phishing_url_alt": probs[3]
    }

    # Determine the most likely classification
    max_label = max(labels.items(), key=lambda x: x[1])

    # Build formatted likely label
    result_str = f"S **Prediction**: '{max_label[0]}' (confidence: **{max_label[1]:.2%}**)\n\n"
    result_str += "F **Full Breakdown:**\n"
    for label, prob in labels.items():
        result_str += f"- {label}: {prob:.2%}\n"

    return result_str

# Set up Gradio interface
demo = gr.Interface (
    fn=predict_email,
    inputs=gr.Textbox(lines=15, placeholder="Paste the email here..."),
    outputs="markdown",
    title="Phishing Email Classifier (DistilBERT)",
    description="Paste an email and this model will predict if it's a phishing attempt. Multilabel classification using a fine-tuned DistilBERT model."
)

# Launch the app
demo.launch(share=False)

# Open in browser manually
webbrowser.open("http://127.0.0.1:7860")
```
