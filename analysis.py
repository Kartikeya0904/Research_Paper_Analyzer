import os
import time
import torch
import fitz  
import nltk
import spacy
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch.nn.functional as F
from transformers import LEDTokenizer, LEDForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nlp = spacy.load("en_core_web_sm")


device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    led_tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
    led_model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384").to(device)
    
    scibert_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    scibert_model = AutoModelForSequenceClassification.from_pretrained(
        "allenai/scibert_scivocab_uncased", 
        num_labels=1
    ).to(device)
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")

# Azure client setup
client = ChatCompletionsClient(
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential("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")
)

def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
        return text
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {str(e)}")

def preprocess_text(text):
    try:
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum() and word not in stopwords.words("english")]
        return " ".join(tokens)
    except Exception as e:
        raise RuntimeError(f"Text preprocessing failed: {str(e)}")

def summarize_text(text):
    try:
        inputs = led_tokenizer(
            text, 
            return_tensors="pt", 
            max_length=16384, 
            truncation=True
        ).to(device)
        
        with torch.no_grad():
            summary_ids = led_model.generate(
                inputs["input_ids"],
                max_length=512,
                num_beams=5,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        return led_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        raise RuntimeError(f"Summarization failed: {str(e)}")

def call_azure_ai(prompt):
    try:
        response = client.complete(
            messages=[
                SystemMessage("You are a research assistant."),
                UserMessage(prompt)
            ],
            model="Llama-3.3-70B-Instruct",
            temperature=0.8,
            max_tokens=2048,
            top_p=0.1
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Azure API Error: {str(e)}"

def generate_paper_insights(text):
    truncated_text = text[:15000]
    prompt = f"Analyze this research paper and provide comprehensive insights:\n{truncated_text}"
    return call_azure_ai(prompt)

def extract_scientific_terms(text):
    truncated_text = text[:15000]
    prompt = f"Extract key scientific terms from this paper along with definitions:\n{truncated_text}"
    return call_azure_ai(prompt)

def generate_score(text):
    try:
        inputs = scibert_tokenizer(
            text.strip().replace("\n", " ").strip()[:4096],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            output = scibert_model(**inputs)
        
        score = torch.sigmoid(output.logits).item()
        return int(score * 100)
    except Exception as e:
        raise RuntimeError(f"Scoring failed: {str(e)}")

def process_pdf(pdf_path):
    try:
        text = extract_text_from_pdf(pdf_path)
        preprocessed_text = preprocess_text(text)
        
        initial_summary = summarize_text(preprocessed_text)
        refined_summary = call_azure_ai(f"Refine and improve this summary:\n{initial_summary}")
        
        return {
            'summary': refined_summary,
            'insights': generate_paper_insights(preprocessed_text),
            'terms': extract_scientific_terms(preprocessed_text),
            'score': generate_score(preprocessed_text)
        }
    except Exception as e:
        raise RuntimeError(f"Processing failed: {str(e)}")