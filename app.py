from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import spacy
import pdfplumber
from rank_match import rank_cvs_with_transformers
from rank_match import rank_cvs_with_spacy_similarity
import logging

app = Flask(__name__)
CORS(app)

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
    model_path = "D:/CVNER/archive/output/model-last/"
    nlp = spacy.load(model_path)
    doc = nlp(cv_text)
    cv_ner_outputs = []
    for ent in doc.ents:
        cv_ner_outputs.append([ent.text])
    return cv_ner_outputs

@app.route('/process_cvs', methods=['POST'])
def process_cvs():
    try:
        uploaded_files = request.files.getlist('files')
        job_description = request.form.get('jobDescription')
        cv_ner_outputs_list = []
        cv_text_list = []

        for uploaded_file in uploaded_files:
            pdf_path = f"uploads/{uploaded_file.filename}"
            txt_path = f"uploads/{os.path.splitext(uploaded_file.filename)[0]}.txt"
            
            uploaded_file.save(pdf_path)
            
            pdf_to_text(pdf_path)
            
            with open(txt_path, "r", encoding='utf-8') as f:
                cv_text = f.read()  
                cv_text_list.append(cv_text)    
            
            # cv_ner_outputs = []
            # for line in cv_text_lines:
            #     processed_line_ner = process_cv_ner(line)
            #     cv_ner_outputs.extend(processed_line_ner)
            # cv_ner_outputs_list.append(cv_ner_outputs)
            cv_ner_outputs = process_cv_ner(cv_text)
            cv_ner_outputs_list.append(cv_ner_outputs)

        app.logger.info('Files uploaded successfully')

        ranked_cvs = rank_cvs_with_spacy_similarity(job_description, cv_text_list)
        ranked_cv_data = []

        for rank, (cv_idx, score) in enumerate(ranked_cvs, start=1):  
            ranked_cv_data.append({
                "rank": rank,
                # "cv_idx": cv_idx, 
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