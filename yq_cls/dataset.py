# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2020-09-20
@LastEditTime: 2020-09-20
@LastEditors: Please set LastEditors
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the dataset used of the emotion classifier.
@All Right Reserve
'''
import pandas as pd
from torch.utils.data import Dataset
import torch


class EmotionDataset(Dataset):
    def __init__(self, paths, tokz, label_vocab, logger, max_lengths=2048):
        # Init attributes
        self.logger = logger
        self.label_vocab = label_vocab
        self.max_lengths = max_lengths
        # Read all the data_1 into memory
        self.data = EmotionDataset.make_dataset(paths, tokz, label_vocab, logger, max_lengths)

    @staticmethod
    def make_dataset(paths, tokz, label_vocab, logger, max_lengths):
        logger.info('reading data_1 from {}'.format(paths))
        dataset = []
        #########################################
        # TODO 2. Read all the data_1 into memory #
        #########################################
        for path in paths:
            # df = pd.read_csv(path, sep='\t', names=['Label', 'Text'])
            df = pd.read_csv(path, names=['Label', 'Text', 'text_split'], header=None)
            for _, row in df.iterrows():
                label = label_vocab[str(row[0])]
                text = row[1]
                inputs = tokz.encode_plus(text, max_length=max_lengths, padding='max_length', truncation=True)
                dataset.append((label, inputs['input_ids'], inputs['attention_mask']))

        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, utt, mask = self.data[idx]
        return {"label": label, "utt": utt, "mask": mask}


class PadBatchSeq:
    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        # Pad the batch with self.pad_id
        res = dict()
        res['label'] = torch.LongTensor([i['label'] for i in batch])
        max_len = max([len(i['utt']) for i in batch])
        res['utt'] = torch.LongTensor([i['utt'] + [self.pad_id] * (max_len - len(i['utt'])) for i in batch])
        res['mask'] = torch.LongTensor([i['mask'] + [self.pad_id] * (max_len - len(i['mask'])) for i in batch])
        return res
