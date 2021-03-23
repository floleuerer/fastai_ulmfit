import re
import glob
import random
import shutil
import json


def split_wiki(path_extract, path_docs, lang):
    name = f'{lang}wiki'
    dest = path_docs
    source = path_extract/'AA'/'wiki_00'
    if dest.exists():
        print(f"{dest} already exists; not splitting")
        return dest
    else:
        print(f'splitting {source} into {dest}')

    dest.mkdir(exist_ok=True, parents=True)
    title_re = re.compile(rf'<doc id="\d+" url="\?curid=\d+" title="([^"]+)">')
    lines = source.open()
    f=None

    for i,l in enumerate(lines):
        if i%100000 == 0: print(i)
        if l.startswith('<doc id="'):
            title = title_re.findall(l)[0].replace('/','_')
            if len(title)>120: continue
            if f: f.close()
            try:
                f = (dest/f'{title}.txt').open('w')
            except Exception as e:
                print('Error:', e)
                f = None
        elif l.startswith('</doc>'):
            continue
        else: 
            if f: f.write(l)
    f.close()
    return dest


def sample_docs(path_docs, path_lm, min_doc_length, number_docs):
    d = str(f'{path_docs}/*.txt')
    files = glob.glob(d)

    random.shuffle(files)

    i = 0
    n_words = 0
    for fi in files:
        with open(fi, 'r') as f:
            doc = f.readlines()
        
        d = ''.join(doc)
        len_doc = len(d)
        
        if len_doc < min_doc_length: 
            continue
        else:
            shutil.copy(fi, path_lm)
            n_words += len(d.split())
            if i%10000 == 0: print(i)
            i += 1
        
        if i >= number_docs: break
            
    return n_words, i

def cleanup_paths(paths):
    for p in paths:
        print(f'removing {p}')
        try:
            shutil.rmtree(p)
        except Exception as e:
            print(f'Error: {e}')

def save_stats(path, n_docs, n_words):
    stats = {
        'n_docs': n_docs,
        'n_words': n_words
    }
    with open(f'{path}/wiki_stats.log', 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)