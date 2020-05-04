export BERT_BASE_DIR=/Users/qiwenlyu/Development/nlp_bert/mobile_bert_config
pytorch_transformers bert \
  $BERT_BASE_DIR/mobilebert_variables.ckpt \
  $BERT_BASE_DIR/bert_config.json \
  $BERT_BASE_DIR/pytorch_model.bin
