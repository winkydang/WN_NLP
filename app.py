from flask import Flask, request, jsonify
import torch
from flask_restful.reqparse import RequestParser
import torchvision.transforms as transforms
from PIL import Image
import io

from torch import softmax
from transformers import BertTokenizer, BertForSequenceClassification

# 创建一个Flask应用
app = Flask(__name__)

# # 加载训练好的PyTorch模型
# model = torch.load('model.pth')
# model.eval()

# 初始化BERT模型和tokenizer
model_name = "bert-base-uncased"  # 使用预训练的BERT模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2分类问题
model.eval()


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 从请求中获取图像文件
        # text = request.files['text']
        parser_ = RequestParser(bundle_errors=True)
        parser_.add_argument('text', type=str)
        args = parser_.parse_args()
        text = args.text


        # 使用tokenizer将文本转换为模型可接受的格式
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

        # 将输入传递给模型以获取情感分类结果  # 使用模型进行预测
        with torch.no_grad():
            outputs = model(**inputs)

        # 获取分类结果
        logits = outputs.logits
        probabilities = softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

        # 根据模型的输出进行情感分类
        if predicted_class == 0:
            sentiment = "Negative"
        else:
            sentiment = "Positive"

        # 返回JSON格式的预测结果
        return jsonify({'sentiment': sentiment})


# 运行Flask应用
if __name__ == '__main__':
    app.run(debug=True)
