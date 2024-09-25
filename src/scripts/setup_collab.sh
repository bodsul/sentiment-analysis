#!/bin/bash

# git clone https://$1@github.com/bodsul/sentiment-analysis.git

#install gcloud and gsutil
sudo apt-get update

sudo apt-get install apt-transport-https ca-certificates gnupg curl

curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

sudo apt-get update && sudo apt-get install google-cloud-cli

mkdir -p sentiment-analysis/data/sentiment_analysis

gsutil cp gs://bode-datasets/archive.zip .

unzip archive.zip -d sentiment-analysis/data/sentiment_analysis

pip install numpy matplotlib scikit-learn pandas nltk tiktoken torch transformers