#!/bin/bash

git clone https://$1@github.com/bodsul/sentiment-analysis.git

mkdir -p sentiment-analysis/data/sentiment_analysis

gsutil cp gs://bode-datasets/archive.zip .

unzip archive.zip -d sentiment-analysis/data/sentiment_analysis