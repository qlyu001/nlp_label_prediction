import pandas as pd
from flair.data import Corpus
from flair.data import TaggedCorpus
#from flair.datasets import ColumnCorpus
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
columns = {0: 'text', 1: 'ner'}
# this is the folder in which train, test and dev files reside
data_folder = '/testdata'
# 2. what tag do we want to predict?
tag_type = 'ner'
# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: TaggedCorpus = NLPTaskDataFetcher.load_column_corpus(data_folder, columns,
                              train_file='train.txt',
                              test_file='test.txt',
                              dev_file='dev.txt')
#corpus: Corpus = ColumnCorpus(data_folder, columns,
                              #train_file='train.txt',
                              #test_file='test.txt',
                              #dev_file='dev.txt')
# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
tagger =  SequenceTagger.load_from_file('best-model.pt')
#tagger =  SequenceTagger.load('/content/drive/My Drive/nlp_task/best-model.pt')
#result, _ =  tagger.evaluate([corpus.test])
#trainer: ModelTrainer = ModelTrainer(tagger, corpus)
#evaluate(model:flair.nn.Model, data_set:List[flair.data.Sentence], eval_mini_batch_size:int=32, embeddings_in_memory:bool=True, out_path:pathlib.Path=None) -> (<class 'dict'>, <class 'float'>)
#print(result.detailed_results)
#result, score = tagger.evaluate(DataLoader(corpus.test, batch_size=1), out_path=f"predictions.txt")
#print(result.log_line)
trainer: ModelTrainer = ModelTrainer(tagger, corpus)
#help(ModelTrainer)
#https://github.com/flairNLP/flair/issues/617
#print(ModelTrainer.evaluate(tagger, corpus.test))
print("loading file /content/drive/My Drive/nlp_task/best-model.pt")
test_metric, test_loss = ModelTrainer.evaluate(tagger, corpus.test)
print(f'MICRO_AVG: acc {test_metric.micro_avg_accuracy()} - f1-score {test_metric.micro_avg_f_score()}')
print(f'MACRO_AVG: acc {test_metric.macro_avg_accuracy()} - f1-score {test_metric.macro_avg_f_score()}')
for class_name in test_metric.get_classes():
    print(f'{class_name:<10} tp: {test_metric.get_tp(class_name)} - fp: {test_metric.get_fp(class_name)} - '
                     f'fn: {test_metric.get_fn(class_name)} - tn: {test_metric.get_tn(class_name)} - precision: '
                     f'{test_metric.precision(class_name):.4f} - recall: {test_metric.recall(class_name):.4f} - '
                     f'accuracy: {test_metric.accuracy(class_name):.4f} - f1-score: '
                     f'{test_metric.f_score(class_name):.4f}')