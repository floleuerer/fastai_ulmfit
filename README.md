# fast.ai ULMFiT with SentencePiece from pretraining to deployment



**Motivation:**
Why even bother with a non-BERT / Transformer language model? Short answer: you can train a state of the art text classifier with ULMFiT with limited data and affordable hardware. The whole process (preparing the Wikipedia dump, pretrain the language model, fine tune the language model and training the classifier) takes about 5 hours on my workstation with a RTX 3090. The training of the model with FP16 requires less than 8 GB VRAM - so you can train the model on affordable GPUs.

I also saw this paper on the roadmap for fast.ai 2.3 [Single Headed Attention RNN: Stop Thinking With Your Head](https://arxiv.org/abs/1911.11423) which could improve the performance further. 

This Repo is based on: 
- https://github.com/fastai/fastai
- [ULMFiT Paper](https://arxiv.org/abs/1801.06146)
- the fast.ai NLP-course repository: https://github.com/fastai/course-nlp

## Pretrained models

| Language | (local) | code  | Perplexity | Vocab Size | Tokenizer | Download (.tgz files) |
|---|---|---|---|---|---|---|
| German | Deutsch  | de  | 16.1 | 15k | SP | https://bit.ly/ulmfit-dewiki |
| German | Deutsch  | de  | 18.5 | 30k | SP | https://bit.ly/ulmfit-dewiki-30k |
| Dutch | Nederlands | nl  | 20.5  | 15k | SP | https://bit.ly/ulmfit-nlwiki |
| Russian | Русский | ru  | 29.8  | 15k | SP | https://bit.ly/ulmfit-ruwiki |
| Portuguese | Português | pt  | 17.3 | 15k | SP | https://bit.ly/ulmfit-ptwiki |
| Vietnamese | Tiếng Việt | vi  | 18.8 | 15k | SP | https://bit.ly/ulmfit-viwiki |
| Japanese | 日本語 | ja  | 42.6 | 15k | SP | https://bit.ly/ulmfit-jawiki |
| Italian | Italiano | it  | 23.7 | 15k | SP |https://bit.ly/ulmfit-itwiki |
| Spanish | Español | es  | 21.9 | 15k | SP |https://bit.ly/ulmfit-eswiki |
| Korean | 한국어 | ko  | 39.6 | 15k | SP |https://bit.ly/ulmfit-kowiki |
| Thai | ไทย | th  | 56.4 | 15k | SP |https://bit.ly/ulmfit-thwiki |
| Hebrew | עברית | he  | 46.3 | 15k | SP |https://bit.ly/ulmfit-hewiki |
| Arabic | العربية | ar  | 50.0 | 15k | SP |https://bit.ly/ulmfit-arwiki |
| Mongolian | Монгол | mn | | | | see: [Github: RobertRitz](https://github.com/robertritz/NLP/tree/main/02_mongolian_language_model) |


**Download with wget**
````
# to preserve the filenames (.tgz!) when downloading with wget use --content-disposition
wget --content-disposition https://bit.ly/ulmfit-dewiki 
````

### Usage of pretrained models - library fastai_ulmfit.pretrained

I've written a small library around this repo, to easily use the pretrained models. You don't have to bother with model, vocab and tokenizer files and paths - the following functions will take care of that. 

Tutorial:  [fastai_ulmfit_pretrained_usage.ipynb](https://github.com/floleuerer/fastai_ulmfit/blob/main/fastai_ulmfit_pretrained_usage.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/floleuerer/fastai_ulmfit/blob/main/fastai_ulmfit_pretrained_usage.ipynb)


**Installation**
````
pip install fastai-ulmfit
````

**Usage**

```
# import
from fastai_ulmfit.pretrained import *

url = 'http://bit.ly/ulmfit-dewiki'

# get tokenizer - if pretrained=True, the SentencePiece Model used for language model pretraining will be used. Default: False 
tok = tokenizer_from_pretrained(url, pretrained=False)

# get language model learner for fine-tuning
learn = language_model_from_pretrained(dls, url=url, drop_mult=0.5).to_fp16()

# save fine-tuned model for classification
path = learn.save_lm('tmp/test_lm')

# get text classifier learner from fine-tuned model
learn = text_classifier_from_lm(dls, path=path, metrics=[accuracy]).to_fp16()
````

### Extract Sentence Embeddings

```
from fastai_ulmfit.embeddings import SentenceEmbeddingCallback

se = SentenceEmbeddingCallback(pool_mode='concat')
_ = learn.get_preds(cbs=[se])

feat = se.feat
pca = PCA(n_components=2)
pca.fit(feat['vec'])
coords = pca.transform(feat['vec'])
```

## Model pretraining 

### Setup 

#### Python environment

```
fastai-2.2.7
fastcore-1.3.19
sentencepiece-0.1.95
fastinference-0.0.36
```

**Install packages**
`pip install -r requirements.txt`

The trained language models are compatible with other fastai versions!

#### Docker

The Wikipedia-dump preprocessing requires docker https://docs.docker.com/get-docker/.

### Project structure

````
.
├── we                         Docker image for the preperation of the Wikipedia-dump / wikiextractor
└── data          
    └── {language-code}wiki         
        ├── dump                    downloaded Wikipedia dump
        │   └── extract             extracted wikipedia-articles using wikiextractor
        ├── docs 
        │   ├── all                 all extracted Wikipedia articles as single txt-files
        │   ├── sampled             sampled Wikipedia articles for language model pretraining
        │   └── sampled_tok         cached tokenized sampled articles - created by fastai / sentencepiece
        └── model 
            ├── lm                  language model trained in step 2
            │   ├── fwd             forward model
            │   ├── bwd             backwards model
            │   └── spm             SentencePiece model
            │
            ├── ft                  fine tuned model trained in step 3
            │   ├── fwd             forward model
            │   ├── bwd             backwards model
            │   └── spm             SentencePiece model
            │
            └── class               classifier trained in step 4
                ├── fwd             forward learner
                └── bwd             backwards learner
````

### 1. Prepare Wikipedia-dump for pretraining

ULMFiT can be peretrained on relativly small datasets - 100 million tokens are sufficient to get state-of-the art classification results (compared to Transformer models as BERT, which need huge amounts of training data). The easiest way is to pretrain a language model on Wikipedia.

The code for the preperation steps is heavily inspired by / copied from the **fast.ai NLP-course**: https://github.com/fastai/course-nlp/blob/master/nlputils.py

I built a docker container and script, that automates the following steps:
1) Download Wikipedia XML-dump
2) Extract the text from the dump
3) Sample 160.000 documents with a minimum length of 1800 characters (results in 100m-120m tokens) both parameters can be changed - see the usage below

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
usage: preprocess.py [-h] -l LANG [-n NUMBER_DOCS] [-m MIN_DOC_LENGTH] [--mirror MIRROR] [--cleanup]

# To cleanup indermediate files (wikiextractor and all splitted documents) run the following command. 
# The Wikipedia-XML-Dump and the sampled docs will not be deleted!
docker run -v $(pwd)/data:/data -it wikiextractor -l <language-code> --cleanup
```

### 2. Language model pretraining on Wikipedia Dump

Notebook: `2_ulmfit_lm_pretraining.ipynb`

To get the best result, you can train two seperate language models - a forward and a backward model. You'll have to run the complete notebook twice and set the `backwards` parameter accordingly. The models will be saved in seperate folders (fwd / bwd). The same applies to fine-tuning and training of the classifier.

#### Parameters 

Change the following parameters according to your needs:
```
lang = 'de' # language of the Wikipedia-Dump
backwards = False # Train backwards model? Default: False for forward model
bs=128 # batch size
vocab_sz = 15000 # vocab size - 15k / 30k work fine with sentence piece
num_workers=18 # num_workers for the dataloaders
step = 'lm' # language model - don't change
```
#### Training Logs + config

`model.json` contains the **parameters** the language model was trained with and the **statistics** (looses and metrics) of the last epoch 
```json
{
    "lang": "de",
    "step": "lm",
    "backwards": false,
    "batch_size": 128,
    "vocab_size": 15000,
    "lr": 0.01,
    "num_epochs": 10,
    "drop_mult": 0.5,
    "stats": {
        "train_loss": 2.894167184829712,
        "valid_loss": 2.7784812450408936,
        "accuracy": 0.46221256256103516,
        "perplexity": 16.094558715820312
    }
}
```

`history.csv` log of the training metrics (epochs, losses, accuracy, perplexity)
```
epoch,train_loss,valid_loss,accuracy,perplexity,time
0,3.375441551208496,3.369227886199951,0.3934227228164673,29.05608367919922,23:00
...
9,2.894167184829712,2.7784812450408936,0.46221256256103516,16.094558715820312,22:44
```

### 3. Language model fine-tuning on unlabled data

Notebook: `3_ulmfit_lm_finetuning.ipynb`

To improve the performance on the downstream-task, the language model should be fine-tuned. We are using a Twitter dataset (GermEval2018/2019), so we fine-tune the LM on unlabled tweets.

To use the notebook on your own dataset, create a `.csv`-file containing your (unlabled) data in the `text` column.

Files required from the Language Model (previous step):
- Model (*model.pth)
- Vocab (*vocab.pkl)

I am not reusing the SentencePiece-Model from the language model! This could lead to slightly different tokenization but fast.ai (-> language_model_learner()) and the fine-tuning takes care of adding and training unknown tokens! This approach gave slightly better results than reusing the SP-Model from the language model.



### 4. Train the classifier

Notebook: `4_ulmfit_train_classifier.ipynb`

The (fine-tuned) language model now can be used to train a classifier on a (small) labled dataset.

To use the notebook on your own dataset, create a `.csv`-file containing your texts in the `text` and labels in the `label` column.

Files required from the fine-tuned LM (previous step):
- Encoder (*encoder.pth)
- Vocab (*vocab.pkl)
- SentencePiece-Model (spm/spm.model)

### 5. Use the classifier for predictions / inference on new data

Notebook: `5_ulmfit_inference.ipynb`

## Evaluation

### German pretrained model
Results with an ensemble of forward + backward model (see the inference notebook). Neither the fine-tuning of the LM, nor the training of the classifier was optimized - so there is still room for improvement.

Official results: https://ids-pub.bsz-bw.de/frontdoor/deliver/index/docId/9319/file/Struss_etal._Overview_of_GermEval_task_2_2019.pdf

#### Task 1 Coarse Classification 

Classes: OTHER, OFFENSE

Accuracy: 79,68 
F1: 75,96 (best BERT 76,95)

#### Task 2 Fine Classification 

Classes: OTHER, PROFANITY, INSULT, ABUSE

Accuracy: 74,56 %
F1: 52,54 (best BERT 53.59)

### Dutch model

Compared result with: https://arxiv.org/pdf/1912.09582.pdf  
Dataset https://github.com/benjaminvdb/DBRD

Accuracy 93,97 % (best BERT 93,0 %)	

### Japanese model
Copared results with: 
- https://github.com/laboroai/Laboro-BERT-Japanese
- https://github.com/yoheikikuta/bert-japanese  
  
Livedoor news corpus   
Accuracy 97,1% (best BERT ~98 %)

### Korean model

Compared with: https://github.com/namdori61/BERT-Korean-Classification
Dataset: https://github.com/e9t/nsmc
Accuracy 89,6 % (best BERT 90,1 %)

## Deployment as REST-API

see https://github.com/floleuerer/fastai-docker-deploy

.
