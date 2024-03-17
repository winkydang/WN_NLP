# -*- coding: utf-8 -*-
'''
@Author: silver
@Date: 2020-09-20
@LastEditTime: 2020-09-20
@LastEditors: Please set LastEditors
@Description: This file is for the NLP capstone of GreedyAI.com. 
    This file implement the model of the emotion classifier.
@All Right Reserve
'''

from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss


class EmotionCLS(BertPreTrainedModel):
    def __init__(self, config):
        # Init the super class
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        # Init the bert encoder    # 初始化bert模型
        self.bert = BertModel(config)
        # classifier_dropout = (config.hidden_dropout_prob)
        classifier_dropout = (
            config.hidden_dropout_prob if config.hidden_dropout_prob is not None else 0.1
        )
        # Init the dropout layer   初始化dropout层
        self.dropout = nn.Dropout(classifier_dropout)
        # Init the classification head    初始化分类层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Init other weights  初始化其他权重
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        """Define forward pass for the model.
        Args:
            input_ids: input token ids. `torch.LongTensor` of shape [batch_size, sequence_length] 
            attention_mask: mask for input ids. `torch.LongTensor` of shape [batch_size, sequence_length] 
            token_type_ids: token types for input token ids. `torch.LongTensor` of shape [batch_size, sequence_length]
            labels: labels for the input token ids. `torch.LongTensor` of shape [batch_size, sequence_length]
        Returns:
            loss: the loss value of the model. loss=None if labels==None
            logits: logits for the classification `torch.FloatTensor` of shape [batch_size, num_labels].
        """
        loss, logits = None, None

        #################################################
        # 3. TODO - Add the code to do the forward pass #
        #################################################
        # step1：传递输入到BERT模型
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)

        # BERT模型的输出是一个包含多个项的元组，其中第一个项是sequence_output
        # sequence_output对应于每个token的表示，通常取第一个token（[CLS] token）来做分类
        sequence_output = outputs[0]
        cls_output = sequence_output[:, 0, :]  # 获取[CLS]的输出

        # # 获取序列输出中[CLS] token的表示      # 来自BERT模型中的pooler层，它应用在最后一层的[CLS] token上，可以用于分类任务。
        # pooled_output = outputs.pooler_output
        # cls_output = outputs.pooler_output

        # step2：在BERT的输出上应用dropout      Dropout可以作为一种正则化方法来避免过拟合
        cls_output = self.dropout(cls_output)

        # step3: 传递dropout后的输出到 分类层
        logits = self.classifier(cls_output)

        # step4: 计算损失（如果提供了Labels）
        loss = None
        if labels is not None:
            # 通常使用交叉熵损失函数来做分类任务
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 返回损失和logits
        return loss, logits
