import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import time 
start = time.time() 

data = pd.read_csv("../testdata/train_stanford.csv", encoding="latin1").fillna(method="ffill")
print(data.tail(10))


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
print(sentences[0])

labels = [[s[2] for s in sent] for sent in getter.sentences]
print(labels[0])

tags_vals = list(set(data["Tag"].values))
tag2idx = {t: i for i, t in enumerate(tags_vals)}



import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
#from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam

import matplotlib.pyplot as plt
import seaborn as sns

# Use plot styling from seaborn.
def plotG( loss_values , validation_loss_values ):  
    sns.set(style='darkgrid')
    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o', label="training loss")
    plt.plot(validation_loss_values, 'r-o', label="validation loss")

        # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

MAX_LEN = 40
bs = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

print(torch.cuda.get_device_name(0))
tokenizer =  BertTokenizer.from_pretrained('./config/uncased_L-24_H-128_B-512_A-4_F-4_OPT.json', do_lower_case=True)
#BERT_FP = './config/uncased_L-24_H-1024_B-512_A-4.json'
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
print(tokenized_texts[0])