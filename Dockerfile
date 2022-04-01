FROM python:3.9-buster

LABEL org.opencontainers.image.authors="deetsadi"
LABEL version="1.0"
LABEL description="This is an intelligent Twitter Bot that uses Python's tweepy and NLP to retweet news/tips related to machine learning, data science, and AI."

COPY bot/main.py /bot/
COPY bot/predict.py /bot/
COPY requirements.txt /tmp
COPY resources/application.yaml /resources/
COPY model/vocab2index.json /model/

RUN apt-get update && \
    apt-get -y install sudo git gcc

RUN pip install -r /tmp/requirements.txt

WORKDIR /bot
EXPOSE 8000:8000
CMD ["python", "main.py"]