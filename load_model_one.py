from transformers import BertTokenizer, BertConfig, RobertaTokenizer
from transformers import BertForTokenClassification, RobertaForTokenClassification
import torch
import numpy as np
import pickle

f = open('tags_vals.pckl', 'rb')
tags_vals = pickle.load(f)
f.close()
print(tags_vals)

f = open('tag2idx.pckl', 'rb')
tag2idx = pickle.load(f)
f.close()
idx2tag = dict((v,k) for k, v in tag2idx.items())

device = torch.device("cpu")
output_dir = './roberta_few_labels/'
tokenizer =  RobertaTokenizer.from_pretrained(output_dir)
model = RobertaForTokenClassification.from_pretrained(output_dir)
model.to(device)


text = 'O, B-Diagnostic_procedure, I-Diagnostic_procedure,B-Biological_structure, I-Biological_structure, B-Sign_symptom, I-Sign_symptom, B-Detailed_description, I-Detailed_description, B-Lab_value, I-Lab_value, B-Date, I-Date, B-Age, I-Age, B-Clinical_event, I-Clinical_event, B-Date, I-Date, B-Disease_disorder, I-Disease_disorder, B-Nonbiological_location, I-Nonbiological_location, B-Severity, I-Severity, B-Sex, B-Therapeutic_procedure, I-Therapeutic_procedure'
tag_values =  text.split(',')

print(tag_values)


query = "a woman aged 65 has a fever and a cough on march at a hospital"
tokenized_sentence = tokenizer.encode(query)
print(tokenized_sentence)
input_ids = torch.tensor([tokenized_sentence]).to(device)
print(input_ids)
predictions = []
with torch.no_grad():
    output = model(input_ids)
    #label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
    output = output[0].detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(output, axis=2)])
    print(predictions)

for x in predictions[0]:
    print(idx2tag[int(x)])


"""
tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
new_tokens, new_labels = [], []
for token, label_idx in zip(tokens, label_indices[0]):
    if token.startswith("##"):
        new_tokens[-1] = new_tokens[-1] + token[2:]
    else:
        new_labels.append(tag_values[label_idx])
        new_tokens.append(token)

print(new_labels)
print(new_tokens)
"""