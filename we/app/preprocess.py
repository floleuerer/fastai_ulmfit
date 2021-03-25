import os
import sys
import json
from pathlib import Path
import shutil
import argparse

from wikiutils import split_wiki, sample_docs, cleanup_paths, save_stats


parser = argparse.ArgumentParser()

parser.add_argument("-l", "--lang", required=True, help="language of the wiki-dump to download / preprocess", type=str)
parser.add_argument("-n", "--number_docs", default=160000, help="number of documents to randomly sample into docs_lm", type=int)
parser.add_argument("-m", "--min_doc_length", default=1800, help="minimum document length (characters) of sampled documents", type=int)
parser.add_argument("--mirror", default='dumps.wikimedia.org', help="wikipedia mirror - default dumps.wikimedia.org", type=str)
parser.add_argument('--cleanup', default=False, action='store_true')
args = parser.parse_args()

lang = args.lang
min_doc_length = args.min_doc_length
number_docs = args.number_docs
mirror = args.mirror
cleanup = args.cleanup

name = f'{lang}wiki'
xml_name = f'{name}-latest-pages-articles.xml'
zip_name = f'{xml_name}.bz2'
path_data = Path('/data')
path_wiki = path_data/name
path_docs = path_data/name/'docs'/'all'
path_dump = path_data/name/'dump'
path_extract = path_dump/'extract'
path_lm = path_data/name/'docs'/'sampled'


# CLEANUP - if --cleanup is set remove directorys and EXIT!
if cleanup:
    paths = [path_docs, path_extract]
    cleanup_paths(paths)
    sys.exit(f'Exiting after clean up!')


if not path_wiki.exists():
    os.mkdir(path_wiki)
    os.mkdir(path_wiki/'docs')

if not path_data.exists():
    os.mkdir(path_data)
else:
    print(f'{path_data} already exists')

# DOWNLOAD wikipedia dump

if not (path_dump/xml_name).exists():
    if not path_dump.exists():
        print(f'creating {path_dump}')
        os.mkdir(path_dump)
    # xml does not exist -> download?
    if not (path_dump/zip_name).exists():
        print(f'downloading {zip_name}')
        # zip does not exist -> download!
        os.system(f'wget --no-check-certificate -P {path_dump.absolute()}/ https://{mirror}/{name}/latest/{name}-latest-pages-articles.xml.bz2')

    print(f'unpacking {zip_name}')
    os.system(f'bzip2 -d {path_dump}/{zip_name}')
    
# EXTRACT with wikiextractor

if not path_extract.exists():
    print(f'creating {path_extract}')
    print('running wikiextractor')
    os.system(f'wikiextractor --no-templates -b 100G -q -o /data/{name}/dump/extract/ /data/{name}/dump/{name}-latest-pages-articles.xml')
else:
    print(f'{path_extract} exists - not extracting')


# SPLIT wiki

print('splitting wiki')
if not path_docs.exists():
    print(f'creating {path_docs}')
    split_wiki(path_extract, path_docs, lang)
else:
    print(f'path {path_docs} exists - not splitting')


# SAMPLE n-docs 

n_words = 0
n_docs = 0
print(f'sampling {number_docs} docs')
if not path_lm.exists():
    print(f'creating {path_lm}')
    os.mkdir(path_lm)
    n_words, n_docs = sample_docs(path_docs, path_lm, min_doc_length, number_docs)
else: 
    print(f'{path_lm} exists - skipping sampling documents!')

save_stats(path_wiki/'docs', n_docs, n_words, min_doc_length)

print(f'sucessfully prepared {name} - {path_lm}, number of docs {n_docs}/{number_docs} with {n_words} words / tokens!')
