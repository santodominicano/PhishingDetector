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
