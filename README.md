# fast.ai ULMFiT with SentencePiece from pretraining to deployment

**Motivation:**
Why even bother with a non-BERT / Transformer languag model? Short answer: you can train a state of the art text classifier with ULMFiT with limited data and affordable hardware. The whole process (preparing the Wikipedia dump, pretrain the language model, fine tune the language model and training the classifier) takes about 5 hours on my workstation with a RTX 3090. The training of the model with FP16 requires less than 8 GB VRAM - so you can train the model on affordable GPUs.

I also saw this paper on the roadmap for fast.ai 2.3 [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423) which could improve the performance further. 

This Repo is based on the fast.ai NLP-course repository: https://github.com/fastai/course-nlp

## Pretrained models

**German / Deutsch**  
Vocab size 15000 (SentencePiece tokenizer) trained on 160k German Wikipedia-Articles (forward + backward model)
https://tinyurl.com/ulmfit-dewiki

## Setup 

### Python environment

**Tested with**
```
fastai-2.2.7
fastcore-1.3.19
sentencepiece-0.1.95
fastinference-0.0.36
```

**Install packages**
`pip install -r requirements.txt`

The trained language models are compatible with other fastai versions!

### Docker

The Wikipedia-dump preprocessing requires docker https://docs.docker.com/get-docker/.


## Project structure
- /we (Docker image for the preperation of the Wikipedia-dump / wikiextractor)
- /data 
  - /{language-code}wiki (created during preperation)
    - /dump (downloaded Wikipedia dump)
      - extract (extract text using wikiextractor)
    - /docs 
      - /all (all extracted Wikipedia articles as txt-files)
      - /sampled (sampled Wikipedia articles for language model pretraining)
    - /model
      - lm (language model trained in step 2)
      - ft (fine tuned model trained in step 3)
      - class (classifier trained in step 4)

# Pretraining, Fine-Tuning and training of the Classifier 

## 1. Prepare Wikipedia-dump for pretraining

ULMFiT can be peretrained on relativly small datasets - 100 million tokens are sufficient to get state-of-the art classification results (compared to Transformer models as BERT, which need hughe amounts of training data). The easiest way is to pretrain a language model on Wikipedia.

The code for the preperation steps is heavily inspired by the fast.ai NLP Course: https://github.com/fastai/course-nlp/blob/master/nlputils.py

I built a docker container and script, that automates the following steps:
1) Download Wikipedia XML-dump
2) Extract the text from the dump
3) Sample 160.000 documents with a minimum length of 1800 characters (results in 100m-120m tokens)

The whole process will take some time depending on the download speed and your hardware. For the 'dewiki' the preperation took about 45 min.

Run the following commands in the current directory
```
# build the wikiextractor docker file
docker build -t wikiextractor ./we

# run the docker container for a specific language
# docker run -v $(pwd)/data:/data -it wikiextractor -l <language-code> 
# for German language-code de run:
docker run -v $(pwd)/data:/data -it wikiextractor -l de
...
sucessfully prepared dewiki - /data/dewiki/docs/sampled, number of docs 160000/160000 with 110699119 words / tokens!

# To change the number of sampled documents or the minimum length see
usage: preprocess.py [-h] -l LANG [-n NUMBER_DOCS] [-m MIN_DOC_LENGTH] [--mirror MIRROR]
```

The Docker image will create the following folders

## 2. Language model pretraining on Wikipedia Dump

Notebook: `2_ulmfit_lm_pretraining.ipynb`

To get the best result, you can train two seperate language models - a forward and a backward model. You'll have to run the complete notebook twice and set the `backwards` parameter accordingly. The models will be saved in seperate folders (fwd / bwd). The same applies to fine-tuning and training of the classifier.

Change the following parameters according to your needs:
```
lang = 'de' # language of the Wikipedia-Dump
backwards = False # Train backwards model? Default: False for forward model
bs=128 # batch size
vocab_sz = 15000 # vocab size - 15k / 30k work fine with sentence piece
num_workers=18 # num_workers for the dataloaders
step = 'lm' # language model - don't change
```

## 3. Language model fine-tuning on unlabled data

Notebook: `3_ulmfit_lm_finetuning.ipynb`

To improve the performance on the downstream-task, the language model should be fine-tuned. We are using a Twitter dataset (GermEval2018/2019), so we fine-tune the LM on unlabled tweets.

## 4. Train the classifier

Notebook: `4_ulmfit_train_classifier.ipynb`

The (fine-tuned) language model now can be used to train a classifier on a small labled dataset. 

## 5. Use the classifier for predictions / inference on new data

Notebook: `5_ulmfit_inference.ipynb`

# Results on GermEval2019

Results with an ensemble of forward + backward model (see the inference notebook). Neither the fine-tuning of the LM, nor the training of the classifier was optimized - so there is still room for improvement.

Official results: https://ids-pub.bsz-bw.de/frontdoor/deliver/index/docId/9319/file/Struss_etal._Overview_of_GermEval_task_2_2019.pdf

## Task 1 Coarse Classification 

Classes: OTHER, OFFENSE

Accuracy: 79,68 
F1: 75,96 (best BERT 76,95)

## Task 2 Fine Classification 

Classes: OTHER, OFFENSE

Accuracy: 74,56 %
F1: 52,54 (best BERT 53.59)

# Deployment as REST-API

see https://github.com/floleuerer/fastai-docker-deploy
