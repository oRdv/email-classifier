from flask import Flask, render_template, request, jsonify, send_from_directory
from email_processor import extract_text_from_pdf, preprocess_text, classify_email, generate_response
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/classify', methods=['POST'])
def classify_email_route():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename.endswith('.pdf'):
                email_text = extract_text_from_pdf(file)
            elif file.filename.endswith('.txt'):
                email_text = file.read().decode('utf-8')
            else:
                return jsonify({'error': 'Formato de arquivo não suportado'})
        else:
            email_text = request.form['emailText']
        
        if not email_text.strip():
            return jsonify({'error': 'Nenhum texto fornecido'})
        
        processed_text = preprocess_text(email_text)
        
        classification_result = classify_email(processed_text)
        category = classification_result['category']
        confidence = classification_result['confidence']
        
        response_text = generate_response(processed_text, category)
        
        return jsonify({
            'category': category,
            'confidence': float(confidence),
            'response': response_text,
            'processed_text': processed_text
        })
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)