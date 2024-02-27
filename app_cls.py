import numpy as np
from flask import Flask, request, jsonify
import torch
from flask_restful.reqparse import RequestParser
import torchvision.transforms as transforms
from PIL import Image
import io

from torch import softmax
from transformers import BertTokenizer, BertForSequenceClassification

# 创建一个Flask应用
app_cls = Flask(__name__)

# # 初始化BERT模型和tokenizer
# model_name = "bert-base-uncased"  # 使用预训练的BERT模型
# tokenizer = BertTokenizer.from_pretrained(model_name)
# model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2分类问题
# model.eval()

# 初始化BERT模型和tokenizer
model_path = './bert-finetuned-sem_eval-english/checkpoint-4275'
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # 初始化一个分词器，来处理输入文本
tokenizer = BertTokenizer.from_pretrained(model_path)  # 初始化一个分词器，来处理输入文本
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

id2label = {0: 'anger', 1: 'anticipation', 2: 'disgust', 3: 'fear', 4: 'joy', 5: 'love', 6: 'optimism', 7: 'pessimism', 8: 'sadness', 9: 'surprise', 10: 'trust'}

@app_cls.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        # 从请求中获取图像文件
        # text = request.files['text']
        parser_ = RequestParser(bundle_errors=True)
        parser_.add_argument('text', type=str)
        args = parser_.parse_args()
        text = args.text

        encoding = tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(model.device) for k, v in encoding.items()}

        outputs = model(**encoding)
        logits = outputs.logits
        print(logits.shape)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= 0.5)] = 1
        predicted_labels = [id2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
        print(predicted_labels)

        # 返回JSON格式的预测结果
        return jsonify({'result': predicted_labels})


# 运行Flask应用
if __name__ == '__main__':
    app_cls.run(debug=True, port=8600)
