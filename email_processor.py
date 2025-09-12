from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import PyPDF2
import re
import torch

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

model_name = "pierreguillou/gpt2-small-portuguese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def extract_text_from_pdf(pdf_file):
    """
    Extrai texto de arquivos PDF
    """
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    """
    Pré-processa o texto removendo caracteres especiais e espaços extras
    """
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def classify_email(text):
    """
    Classifica o email em produtivo ou improdutivo
    """
    candidate_labels = ["produtivo", "improdutivo"]
    classification_result = classifier(text, candidate_labels)
    return {
        'category': classification_result['labels'][0],
        'confidence': classification_result['scores'][0]
    }

def generate_response(text, category):
    """
    Gera uma resposta automática baseada na categoria do email usando IA
    """
    if category == "produtivo":
        prompt = f"Como assistente virtual de uma empresa financeira, responda de forma profissional e útil ao seguinte email: '{text}'\nResposta:"
    else:
        prompt = f"Como assistente virtual, responda de forma educada e breve ao seguinte email: '{text}'\nResposta:"
    
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=300, 
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = response.replace(prompt, "").strip()
    
    if not response.endswith('.'):
        response += '.'
    
    return response