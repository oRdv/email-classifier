from transformers import pipeline
import PyPDF2
import re
import numpy as np

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli")

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
    Classifica o email em produtivo ou improdutivo com regras mais rigorosas
    """
    unproductive_keywords = [
        'oi galerinha', 'feliz', 'natal', 'ano novo', 'páscoa', 'feriado',
        'fim de semana', 'boa sorte', 'parabéns', 'comemoração', 'festa',
        'agradecimento', 'saudações', 'cumprimentos', 'abraços', 'beijos',
        'alegria', 'comemorar', 'desejo', 'felicitações', 'animado', 'celebração',
        'bom dia', 'boa tarde', 'boa noite', 'saudades', 'mensagem carinhosa'
    ]
    
    text_lower = text.lower()
    is_unproductive = any(keyword in text_lower for keyword in unproductive_keywords)
    
    if is_unproductive:
        return {
            'category': "improdutivo",
            'confidence': 0.95  
        }
    
    candidate_labels = ["produtivo", "improdutivo"]
    classification_result = classifier(text, candidate_labels)
    
    category = classification_result['labels'][0]
    confidence = classification_result['scores'][0]
    
    if category == "improdutivo":
        confidence = max(0.3, confidence * 0.7)
    
    return {
        'category': category,
        'confidence': float(confidence)
    }

def generate_response(text, category):
    """
    Gera uma resposta automática baseada na categoria do email
    """
    if category == "produtivo":
        responses = [
            "Agradecemos seu contato. Nossa equipe analisará sua solicitação e retornará em breve.",
            "Recebemos sua solicitação e estamos processando sua requisição. Retornaremos em até 24 horas.",
            "Obrigado por entrar em contato. Estamos verificando as informações e retornaremos em breve.",
            "Sua solicitação foi recebida e está sendo processada. Aguarde nosso retorno.",
            "Confirmamos o recebimento de sua mensagem. Nossa equipe entrará em contato em breve.",
            "Agradecemos pelo seu email. Estamos analisando sua solicitação e retornaremos em breve.",
            "Sua mensagem foi recebida com sucesso. Nossa equipe está trabalhando em sua solicitação.",
            "Obrigado pelo contato. Estamos processando sua solicitação e retornaremos em breve.",
            "Recebemos sua mensagem e estamos analisando o caso. Retornaremos em breve.",
            "Agradecemos sua solicitação. Nossa equipe está verificando as informações necessárias."
        ]
    else:
        responses = [
            "Agradecemos sua mensagem. Desejamos a você um ótimo dia!",
            "Obrigado pelo contato. Ficamos felizes com sua mensagem!",
            "Agradecemos sua mensagem amigável. Desejamos tudo de bom!",
            "Obrigado por compartilhar suas boas energias conosco!",
            "Agradecemos seu carinho e atenção. Tenha um excelente final de semana!",
            "Obrigado pela mensagem. Desejamos a você uma ótima semana!",
            "Agradecemos suas palavras. Que tenha um dia maravilhoso!",
            "Obrigado pelo carinho. Desejamos a você tudo de melhor!",
            "Agradecemos a mensagem. Que seu dia seja repleto de alegrias!",
            "Obrigado pelas palavras. Desejamos a você sucesso em seus projetos!"
        ]
    
    text_hash = hash(text) % len(responses)
    return responses[text_hash]