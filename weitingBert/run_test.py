import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, RobertaTokenizer
from transformers import BertForTokenClassification, RobertaForTokenClassification
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import time 



import matplotlib.pyplot as plt
import seaborn as sns


from seqeval.metrics import f1_score



data = pd.read_csv("train_stanford4-test.csv", encoding="latin1").fillna(method="ffill")

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)
sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
labels = [[s[2] for s in sent] for sent in getter.sentences]
import pickle
f = open('tags_vals.pckl', 'rb')
tags_vals = pickle.load(f)
f.close()

f = open('tag2idx.pckl', 'rb')
tag2idx = pickle.load(f)
f.close()




MAX_LEN = 40
bs = 8
output_dir = './model_save/'
tokenizer = RobertaTokenizer.from_pretrained(output_dir)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["O"], padding="post",
                     dtype="long", truncating="post")

attention_masks = [[float(i>0) for i in ii] for ii in input_ids]

test_inputs = torch.tensor(input_ids)
test_masks = torch.tensor(attention_masks)
test_tags = torch.tensor(tags)


test_data = TensorDataset(test_inputs, test_masks, test_tags)
test_sampler = RandomSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()



model = RobertaForTokenClassification.from_pretrained(output_dir)
model.cuda()



predictions = []
true_labels = []
eval_loss, eval_accuracy = 0, 0
nb_eval_steps, nb_eval_examples = 0, 0
for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    b_input_ids, b_input_mask, b_labels = batch
    b_input_ids=torch.tensor(b_input_ids).to(torch.int64)
    #print("b_input_ids")
    #print(b_input_ids)
    #print(b_input_ids.shape)
    #print(b_input_mask)
    b_labels = b_labels.long()

    with torch.no_grad():
        tmp_eval_loss, x = model(b_input_ids, token_type_ids=None,
                              attention_mask=b_input_mask, labels=b_labels)
        logits = model(b_input_ids, token_type_ids=None,
                       attention_mask=b_input_mask)
        
    logits = logits[0].detach().cpu().numpy()
    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
    label_ids = b_labels.to('cpu').numpy()
    true_labels.append(label_ids)
    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    eval_loss += tmp_eval_loss.mean().item()
    eval_accuracy += tmp_eval_accuracy

    nb_eval_examples += b_input_ids.size(0)
    nb_eval_steps += 1

pred_tags = [[tags_vals[p_i] for p_i in p] for p in predictions]

valid_tags = [[tags_vals[l_ii] for l_ii in l_i] for l in true_labels for l_i in l ]

#end = time.time()
#hours, rem = divmod(end-start, 3600)
#minutes, seconds = divmod(rem, 60)
#print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

#print("--- %s seconds ---" % (time.time() - start))
#print("Validation loss: {}".format(eval_loss/nb_eval_steps))
print("Test Accuracy: {}".format(eval_accuracy/nb_eval_steps))
print("Test F1-Score: {}".format(f1_score(pred_tags, valid_tags)))



from sklearn.metrics import classification_report, accuracy_score

flat_pred = [item for sublist in pred_tags for item in sublist]
flat_valid = [item for sublist in valid_tags for item in sublist]

print(accuracy_score(flat_valid, flat_pred))
print(classification_report(flat_valid, flat_pred))

"""
tokenized_texts = tokenizer.tokenize("a boy has a fever after ten")
print(tokenized_texts)
MAX_LEN = 40
input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(tokenized_texts)],
                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
print(input_ids[0])


MAX_LEN = 40

# Copy the model to the GPU.
model.to(device)

#input_ids = torch.tensor([input_ids[0],input_ids[0],input_ids[0],input_ids[0],input_ids[0],input_ids[0],input_ids[0],input_ids[0]]).long().unsqueeze(0)  # Batch size 1
input_ids = torch.tensor(tokenizer.encode("a boy has a fever after ten")).long().unsqueeze(0)
#print(input_ids[0].shape)
#b_input_mask = torch.tensor([[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]).long()

print(input_ids.shape)
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
#print(labels)
logits = model(input_ids[0])
#print(len(outputs[0].shape))
logits = logits[0].detach().cpu().numpy()
print(logits)
import numpy as np
predictions = []
predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
print(predictions)
"""