from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import word_tokenize
def tokenize(s):
    s = s.replace('-', ' - ')       # deal with special case 17-year-old
    return ' '.join(word_tokenize(s))
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
def sentenceLabel(sentence):
    model = SequenceTagger.load_from_file('./best-model.pt')
    sentence = [q for q in sentence.split('\\n')]
    # predict
    all_tokens = []
    all_entities = []

    for q in sentence:
        sen = Sentence(tokenize(q))
        text = q.split(' ')
        #print(text)
        model.predict(sen)
        tokens = []
        entity_types = []
        prev = ''
        for t in sen.tokens:
            if prev == '-' or t.text == '-':
                tokens[-1] = tokens[-1]+t.text
                prev = t.text
                print(tokens[-1])
                continue
            token = t.text
            entity = t.tags['ner'].value
            tokens.append(token)
            if(len(entity) < 2):
                entity_types.append(entity)
            else:
                entity_types.append(entity[2:])
                
        
        all_tokens.append(tokens)
        all_entities.append(entity_types)
    #print(all_tokens)
    #print(all_entities)
    return all_tokens,all_entities
    
if __name__ == '__main__':
    f = open("15939911.txt", "r")
    sentences = []
    tokens = []
for x in f:
    all_tokens,all_entities = sentenceLabel(x)
    sentences.append(all_tokens)
    tokens.append(all_entities)
    
print(sentences)
print(tokens)
location = locationIndex(sentences)
#print(location)
