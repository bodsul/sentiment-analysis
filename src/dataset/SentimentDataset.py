from torch.utils.data import Dataset
import torch
import pandas as pd
import os

import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
# print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

VAL_SPLIT_FROM_TRAIN = 0.1
TRAIN_SPLIT_FROM_TRAIN = 1-VAL_SPLIT_FROM_TRAIN

def collate_fn(batch):
    texts = []
    labels = []

    for text, label in batch:
        texts.append(text)
        labels.append(label)
    return torch.stack(texts), torch.stack(labels)

class SentimentDataset(Dataset):
    def __init__(self, split, tokenizer, max_length=None, data_dir='../data/sentiment_analysis', pad_token_id=50256):
        if split=='validation':
            self.split = 'train'
        else:
            self.split = split
    
        df = pd.read_csv(os.path.join(data_dir, f'{self.split}.csv'), encoding='unicode_escape')[['text', 'sentiment']]
        
        if split=='train' or split=='validation':
            df = df.sample(frac=1, random_state=123).reset_index(drop=True)

            df_negative = df.loc[df['sentiment']=='negative']
            df_neutral = df.loc[df['sentiment']=='neutral']
            df_positive = df.loc[df['sentiment']=='positive']

            if split=='train':
                df_negative=df_negative[:int(len(df_negative)*TRAIN_SPLIT_FROM_TRAIN)]
                df_neutral=df_negative[:int(len(df_neutral)*TRAIN_SPLIT_FROM_TRAIN)]
                df_positive=df_positive[:int(len(df_positive)*TRAIN_SPLIT_FROM_TRAIN)]
            else:
                df_negative=df_negative[int(len(df_negative)*TRAIN_SPLIT_FROM_TRAIN):]
                df_neutral=df_negative[int(len(df_neutral)*TRAIN_SPLIT_FROM_TRAIN):]
                df_positive=df_positive[int(len(df_positive)*TRAIN_SPLIT_FROM_TRAIN):]

            df = pd.concat([df_negative, df_neutral, df_positive])
        
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        df['sentiment'] = df['sentiment'].map({"negative": 0, "neutral": 1, "positive": 2})

        self.df = df
        self.encoded_texts = []
        bad_indices = []

        for i in range(len(df)):
            try:
                self.encoded_texts.append(tokenizer.encode(df.loc[i]['text']))
            except TypeError:
                bad_indices.append(i)
                continue
        print(f'Bad Indices: {len(bad_indices)} out of {len(self.df)}')
        self.df.drop(bad_indices)

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
        
        #B
        self.encoded_texts = [
        encoded_text[:self.max_length]
        for encoded_text in self.encoded_texts
        ]
        
        #C
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
        
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.df.iloc[index]["sentiment"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
    )

    def __len__(self):
        return len(self.df)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
        if encoded_length > max_length:
            max_length = encoded_length
        return max_length


if __name__ == '__main__':
    train_dataset = SentimentDataset('train', tokenizer)
    val_dataset = SentimentDataset('validation', tokenizer)
    test_dataset = SentimentDataset('test', tokenizer)

    print(train_dataset.df.head())
    print(val_dataset.df.head())

    for text, label in train_dataset:
        print(text.shape, label.shape)