import codecs
import configparser
import copy
import distutils.util
import glob
import os
import pickle
from pprint import pprint
import random
import shutil
import sys
import time
import warnings
import pkg_resources

import numpy as np
import brat_to_conll
import conll_to_brat
import utils
import utils_nlp

def _get_default_param():
    """
    Get the default parameters.

    """
    param = {'pretrained_model_folder':'./trained_models/conll_2003_en',
             'dataset_text_folder':'./testdata',
             'character_embedding_dimension':25,
             'character_lstm_hidden_state_dimension':25,
             'check_for_digits_replaced_with_zeros':True,
             'check_for_lowercase':True,
             'debug':False,
             'dropout_rate':0.5,
             'experiment_name':'experiment',
             'freeze_token_embeddings':False,
             'gradient_clipping_value':5.0,
             'learning_rate':0.005,
             'load_only_pretrained_token_embeddings':False,
             'load_all_pretrained_token_embeddings':False,
             'main_evaluation_mode':'conll',
             'maximum_number_of_epochs':100,
             'number_of_cpu_threads':8,
             'number_of_gpus':0,
             'optimizer':'sgd',
             'output_folder':'./output',
             'output_scores':False,
             'patience':10,
             'parameters_filepath': os.path.join('.','parameters.ini'),
             'plot_format':'pdf',
             'reload_character_embeddings':True,
             'reload_character_lstm':True,
             'reload_crf':True,
             'reload_feedforward':True,
             'reload_token_embeddings':True,
             'reload_token_lstm':True,
             'remap_unknown_tokens_to_unk':True,
             'spacylanguage':'en',
             'tagging_format':'bio',
             'token_embedding_dimension':100,
             'token_lstm_hidden_state_dimension':100,
             'token_pretrained_embedding_filepath':'./data/word_vectors/glove.6B.100d.txt',
             'tokenizer':'stanford',
             'train_model':True,
             'use_character_lstm':True,
             'use_crf':True,
             'use_pretrained_model':False,
             'verbose':False}

    return param


def get_valid_dataset_filepaths(parameters):
    """
    Get valid filepaths for the datasets.
    """
    dataset_filepaths = {}
    dataset_brat_folders = {}
    print(parameters['dataset_text_folder'])
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_text_folder'], 
            '{0}.txt'.format(dataset_type))
        dataset_brat_folders[dataset_type] = os.path.join(parameters['dataset_text_folder'], 
            dataset_type)
        dataset_compatible_with_brat_filepath = os.path.join(parameters['dataset_text_folder'], 
            '{0}_compatible_with_brat.txt'.format(dataset_type))

        # Conll file exists
        if os.path.isfile(dataset_filepaths[dataset_type]) \
        and os.path.getsize(dataset_filepaths[dataset_type]) > 0:

            # Brat text files exist
            if os.path.exists(dataset_brat_folders[dataset_type]) \
            and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:

                # Check compatibility between conll and brat files
                brat_to_conll.check_brat_annotation_and_text_compatibility(dataset_brat_folders[dataset_type])
                if os.path.exists(dataset_compatible_with_brat_filepath):
                    dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

                conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepaths[dataset_type], 
                    dataset_brat_folders[dataset_type])

            # Brat text files do not exist
            else:

                # Populate brat text and annotation files based on conll file
                conll_to_brat.conll_to_brat(dataset_filepaths[dataset_type], dataset_compatible_with_brat_filepath, 
                    dataset_brat_folders[dataset_type], dataset_brat_folders[dataset_type])
                dataset_filepaths[dataset_type] = dataset_compatible_with_brat_filepath

        # Conll file does not exist
        else:
            # Brat text files exist
            #print(dataset_brat_folders[dataset_type])
            if os.path.exists(dataset_brat_folders[dataset_type]) \
            and len(glob.glob(os.path.join(dataset_brat_folders[dataset_type], '*.txt'))) > 0:
                dataset_filepath_for_tokenizer = os.path.join(parameters['dataset_text_folder'], 
                    '{0}_{1}.txt'.format(dataset_type, parameters['tokenizer']))
                if os.path.exists(dataset_filepath_for_tokenizer):
                    conll_to_brat.check_compatibility_between_conll_and_brat_text(dataset_filepath_for_tokenizer, 
                        dataset_brat_folders[dataset_type])
                else:
                    # Populate conll file based on brat files
                    brat_to_conll.brat_to_conll(dataset_brat_folders[dataset_type], 
                        dataset_filepath_for_tokenizer, parameters['tokenizer'], 
                        parameters['spacylanguage'])
                dataset_filepaths[dataset_type] = dataset_filepath_for_tokenizer

            # Brat text files do not exist
            else:
                del dataset_filepaths[dataset_type]
                del dataset_brat_folders[dataset_type]
                continue

        if parameters['tagging_format'] == 'bioes':
            # Generate conll file with BIOES format
            bioes_filepath = os.path.join(parameters['dataset_text_folder'], 
                '{0}_bioes.txt'.format(utils.get_basename_without_extension(dataset_filepaths[dataset_type])))
            utils_nlp.convert_conll_from_bio_to_bioes(dataset_filepaths[dataset_type], 
                bioes_filepath)
            dataset_filepaths[dataset_type] = bioes_filepath

    return dataset_filepaths, dataset_brat_folders
    
param = _get_default_param()
dataset_filepaths,dataset_brat_folders = get_valid_dataset_filepaths(param)
print(dataset_filepaths)
print(dataset_brat_folders)