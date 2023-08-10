from sentence_transformers import SentenceTransformer, util
import numpy as np
import spacy

def rank_cvs_with_transformers(job_description, cv_ner_outputs_list):
    model = SentenceTransformer('bert-base-nli-mean-tokens')

    job_description_embedding = model.encode(job_description)

    ranked_cvs = []
    for idx, cv in enumerate(cv_ner_outputs_list):
        cv_text = ' '.join([token for token, _ in cv])
        cv_embedding = model.encode(cv_text)

        cosine_similarity = util.pytorch_cos_sim(job_description_embedding, cv_embedding).numpy()[0][0]
        ranked_cvs.append((idx, cosine_similarity))

    ranked_cvs = sorted(ranked_cvs, key=lambda x: x[1], reverse=True)
    return ranked_cvs

def rank_cvs_with_spacy_similarity(job_description, cv_text_list):
    nlp = spacy.load('en_core_web_lg')

    jd_doc = nlp(job_description)

    ranked_cvs = []
    for idx, cv_text in enumerate(cv_text_list):
        cv_doc = nlp(cv_text)

        similarity_score = jd_doc.similarity(cv_doc)
        ranked_cvs.append((idx, similarity_score))

    ranked_cvs = sorted(ranked_cvs, key=lambda x: x[1], reverse=True)
    return ranked_cvs

