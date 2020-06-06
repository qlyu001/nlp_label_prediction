
import numpy as np
from transformers import BertTokenizer, BertConfig, RobertaTokenizer
from transformers import BertForTokenClassification, RobertaForTokenClassification
import torch
import torch.nn.functional as F
import pickle
import codecs
from nltk import word_tokenize

def locationIndex(sentences):
    start = 0
    end = 0
    location = []
    for sentence in sentences:
        for words in sentence:
            for word in words:
                end = start + len(word)
                #print(word)
                location.append([start,end])
                start = end + 1
    return location
    
def tokenize(tokenizer, text: str):
    """ tokenize input"""
    words = word_tokenize(text)
    tokens = []
    valid_positions = []
    for i,word in enumerate(words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i in range(len(token)):
            if i == 0:
                valid_positions.append(1)
            else:
                valid_positions.append(0)
    #print(tokens)
    #print(valid_positions)
    return tokens, valid_positions

    
def sentenceLabel(sentence):
    f = open('./model_save/tag2idx.pckl', 'rb')
    tag2idx = pickle.load(f)
    device = torch.device("cpu")

    output_dir = './model_save/'
    idx2tag = dict((v,k) for k, v in tag2idx.items())
    tokenizer = RobertaTokenizer.from_pretrained(output_dir)
    model = RobertaForTokenClassification.from_pretrained(output_dir)
    model.aux_logits = False
    model.to(device)
    tokenized_sentence = tokenizer.encode(sentence)
    input_ids = torch.tensor([tokenized_sentence]).to(device)



    all_tokens = []
    origin_tokens = sentence.split(' ')
    print(origin_tokens)
    all_entities = []
    entity_types = []
    tokenized_sentence = tokenizer.encode(sentence)
    input_ids = torch.tensor([tokenized_sentence]).to(device)

    predictions = []
    with torch.no_grad():
        output = model(input_ids)
        output = output[0].detach().cpu().numpy()
        predictions.extend([list(p) for p in np.argmax(output, axis=2)])


    tags_predictions = []
    for x in predictions[0]:
        tags_predictions.append(idx2tag[int(x)])

    tokens = []
    count = 0
    
    ### get tokens from ids
    for x in tokenizer.convert_ids_to_tokens(tokenized_sentence):
        if count == 1:
            tokens.append(x)
        elif x[0] == 'Ġ':
            tokens.append(x[1:])
        else:
            tokens.append(x)
        count+=1

    wordIndex = 0
    startIndex = 0
    entityIndex = 0
    entity_types.append(tags_predictions[1:-1])

    for x in tokens[1:-1]:
        entity = entity_types[0][entityIndex]
        entityIndex += 1
        if wordIndex == len(origin_tokens):
            break
        if x in origin_tokens[wordIndex].lower():
            if startIndex == 0:
                all_tokens.append(origin_tokens[wordIndex])
                if(len(entity) < 2):
                    all_entities.append(entity)
                else:
                    all_entities.append(entity[2:])
            startIndex = startIndex + len(x)
            if startIndex  >= len(origin_tokens[wordIndex]):
                wordIndex += 1
                startIndex = 0
            


    print(all_tokens)
    print(all_entities)

    return all_tokens,all_entities
if __name__ == '__main__':
    file = open("15939911.txt", "r")
    sentences = []
    tokens = []
    sentenceLabel("CASE: A 28-year-old previously healthy man presented with a 6-week history of palpitations.")
        
    #print(sentences)
    #print(tokens)
    location = locationIndex(sentences)
    #print(location)

