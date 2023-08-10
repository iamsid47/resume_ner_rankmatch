from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import spacy
import pdfplumber
from rank_match import rank_cvs_with_transformers
from rank_match import rank_cvs_with_spacy_similarity
import logging
import io
from pdfplumber import open as pdf_open

app = Flask(__name__)
CORS(app)

custom_nlp = spacy.load('D:/CVNER/archive/output/model-last/')

# Load English tokenizer, POS tagger, parser, NER, and word vectors
nlp = spacy.load('en_core_web_trf', disable=['parser', 'ner'])
nlp_trf = spacy.load('en_core_web_trf', disable=['parser', 'textcat'])


def pdf_to_text(pdf_path):
    txt_path = f"uploads/{os.path.splitext(os.path.basename(pdf_path))[0]}.txt"

    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    
    with open(txt_path, "w", encoding='utf-8') as txt_file:
        txt_file.write(text)

    return txt_path

def process_cv_ner(cv_text):
    preprocessed_cv_text = preprocess_text(cv_text)  # Preprocess with en_core_web_trf
    doc = custom_nlp(preprocessed_cv_text)  # Use your custom NER model
    cv_ner_outputs = []
    for ent in doc.ents:
        cv_ner_outputs.append([ent.text, ent.label_])  # Extract only entity text
    return cv_ner_outputs

def preprocess_text(text):
    doc = nlp(text)
    clean_tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            clean_tokens.append(token.lemma_)
    return ' '.join(clean_tokens)

def process_job_description_ner(job_description):
    doc = nlp_trf(job_description)
    job_desc_ner_outputs = []
    for ent in doc.ents:
        job_desc_ner_outputs.append([ent.text, ent.label_])
    return job_desc_ner_outputs

@app.route('/process_cvs', methods=['POST'])
def process_cvs():
    try:
        uploaded_files = request.files.getlist('files')
        job_description_file = request.files['jobDescriptionFile']
        job_description_text = ""
        with pdf_open(job_description_file) as pdf:
            for page in pdf.pages:
                job_description_text += page.extract_text()
        
        job_description_ner_outputs = process_job_description_ner(job_description_text)
        job_description_ner_text = ' '.join([entity_text for entity_text, _ in job_description_ner_outputs])

        cv_ner_outputs_list = []
        cv_text_list = []  

        for uploaded_file in uploaded_files:
            pdf_path = f"uploads/{uploaded_file.filename}"
            txt_path = f"uploads/{os.path.splitext(uploaded_file.filename)[0]}.txt"
            
            uploaded_file.save(pdf_path)
            
            pdf_to_text(pdf_path)
            
            with open(txt_path, "r", encoding='utf-8') as f:
                cv_text = f.read()  
                cv_ner_outputs = process_cv_ner(cv_text)
                cv_ner_outputs_list.append(cv_ner_outputs)

        app.logger.info('Files uploaded successfully')

        # ranked_cvs = rank_cvs_with_spacy_similarity(job_description_ner_text, cv_ner_outputs_list)
        # ranked_cv_data = []


        for idx, cv in enumerate(cv_ner_outputs_list):
            cv_text = ' '.join([entity_text for entity_text, _ in cv])
            cv_text_list.append(cv_text)  # Append CV text to the list
            print(f"CV {idx} Entity Texts:", cv_text)
        ranked_cvs = rank_cvs_with_spacy_similarity(job_description_ner_text, cv_text_list)  # Use job_description_ner_text

        ranked_cv_data = []


        for rank, (cv_idx, score) in enumerate(ranked_cvs, start=1):  
            ranked_cv_data.append({
                "rank": rank,
                "score": float(score * 100),
                "file_name": uploaded_files[cv_idx].filename,
                "ner_output": cv_ner_outputs_list[cv_idx]  
            })

        return jsonify({'ranked_cvs': ranked_cv_data}), 200

    except Exception as e:
        app.logger.error(str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    logging.basicConfig(level=logging.DEBUG)

    app.run(debug=True)