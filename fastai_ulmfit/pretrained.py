# AUTOGENERATED! DO NOT EDIT! File to edit: lib_nbs/00_pretrained.ipynb (unless otherwise specified).

__all__ = ['tokenizer_from_pretrained', 'language_model_from_pretrained', 'text_classifier_from_lm']

# Cell
import json
from fastai.text.all import SentencePieceTokenizer, SpacyTokenizer, language_model_learner, \
                            text_classifier_learner, untar_data, Path, patch, \
                            LMLearner, os, pickle, shutil, AWD_LSTM, accuracy, \
                            Perplexity, delegates

# Cell
def _get_config(path):
    with open(path/'model.json', 'r') as f:
        config = json.load(f)
    return config

# Cell
def _get_pretrained_model(url):
    fname = f"{url.split('/')[-1]}.tgz"
    path = untar_data(url, fname=fname, c_key='model')
    return path

# Cell
def _get_direction(backwards):
    return 'bwd' if backwards else 'fwd'

# Cell
def _get_model_files(path, backwards=False):
    direction = _get_direction(backwards)
    config = _get_config(path/direction)
    try:
        model_path = path/direction
        model_file = list(model_path.glob(f'*model.pth'))[0]
        vocab_file = list(model_path.glob(f'*vocab.pkl'))[0]
        fnames = [model_file.absolute(),vocab_file.absolute()]
    except IndexError: print(f'The model in {model_path} is incomplete, download again'); raise
    fnames = [str(f.parent/f.stem) for f in fnames]
    return fnames

# Cell
def tokenizer_from_pretrained(url, pretrained=False, backwards=False, **kwargs):
    path = _get_pretrained_model(url)
    direction = _get_direction(backwards)
    config = _get_config(path/direction)
    sp_model=path/'spm'/'spm.model' if pretrained else None
    if config['tokenizer']['class'] == 'SentencePieceTokenizer':
        tok = SentencePieceTokenizer(**config['tokenizer']['params'], sp_model=sp_model, **kwargs)
    elif config['tokenizer']['class'] == 'SpacyTokenizer':
        tok = SpacyTokenizer(**config['tokenizer']['params'], **kwargs)
    else:
        raise ValueError('Tokenizer not supported')
    return tok

# Cell
@delegates(language_model_learner)
def language_model_from_pretrained(dls, url=None, backwards=False, metrics=None, **kwargs):
    arch = AWD_LSTM # TODO: Read from config
    path = _get_pretrained_model(url)
    fnames = _get_model_files(path)
    metrics = [accuracy, Perplexity()] if metrics == None else metrics
    return language_model_learner(dls,
                                  arch,
                                  pretrained=True,
                                  pretrained_fnames=fnames,
                                  metrics=metrics,
                                  **kwargs)

# Cell
def _get_model_path(learn=None, path=None):
    path = (learn.path/learn.model_dir) if not path else Path(path)
    if not path.exists(): os.makedirs(path, exist_ok=True)
    return path

# Cell
@patch
def save_lm(x:LMLearner, path=None, with_encoder=True):
    path = _get_model_path(x, path)
    x.to_fp32()
    # save model
    x.save((path/'lm_model').absolute(), with_opt=False)

    # save encoder
    if with_encoder:
        x.save_encoder((path/'lm_encoder').absolute())

    # save vocab
    with open((path/'lm_vocab.pkl').absolute(), 'wb') as f:
        pickle.dump(x.dls.vocab, f)

    # save tokenizer if SentencePiece is used
    if isinstance(x.dls.tok, SentencePieceTokenizer):
        # copy SPM if path not spm path
        spm_path = Path(x.dls.tok.cache_dir)
        if path.absolute() != spm_path.absolute():
            target_path = path/'spm'
            if not target_path.exists(): os.makedirs(target_path, exist_ok=True)
            shutil.copyfile(spm_path/'spm.model', target_path/'spm.model')
            shutil.copyfile(spm_path/'spm.vocab', target_path/'spm.vocab')

    return path

# Cell
@delegates(text_classifier_learner)
def text_classifier_from_lm(dls, path=None, backwards=False, **kwargs):
    arch = AWD_LSTM # TODO: Read from config
    path = _get_model_path(path=path)
    learn = text_classifier_learner(dls, arch, pretrained=False, **kwargs)
    learn.load_encoder((path/'lm_encoder').absolute())
    return learn