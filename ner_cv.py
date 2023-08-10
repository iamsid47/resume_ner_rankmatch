import spacy

model = "D:/CVNER/output/model-best/"
nlp = spacy.load(model)

with open("D:/CVNER/test.txt", "r", encoding='utf-8') as f:
    text = f.read()

doc = nlp(text)
for ent in doc.ents:
    print([ent.text, ent.label_])