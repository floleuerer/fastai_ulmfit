FROM python:3.8-slim

RUN apt-get update && apt-get -y install bzip2 wget
RUN pip install wikiextractor==3.0.4 

ADD app/ .

ENTRYPOINT ["python", "-u", "preprocess.py"]
