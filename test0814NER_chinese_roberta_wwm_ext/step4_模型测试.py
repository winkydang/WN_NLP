import traceback

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json


# 解析实体
def post_processing(outputs, text, labels_map):
    _, predicted_labels = torch.max(outputs.logits, dim=2)

    predicted_labels = predicted_labels.detach().cpu().numpy()

    predicted_tags = [labels_map[label_id] for label_id in predicted_labels[0]]

    result = {}
    entity = ""
    type = ""
    for index, word_token in enumerate(text):
        try:
            if len(text) > 290:
                text = text[:290]

            tag = predicted_tags[index]
            if tag.startswith("B-"):
                type = tag.split("-")[1]
                if entity:
                    if type not in result:
                        result[type] = []
                    result[type].append(entity)
                entity = word_token
            elif tag.startswith("I-"):
                type = tag.split("-")[1]
                if entity:
                    entity += word_token
            else:
                if entity:
                    if type not in result:
                        result[type] = []
                    result[type].append(entity)
                entity = ""
        except Exception as e:
            print(e, traceback.format_exc())
            entity = ""
            continue
    return result


def main():
    labels_path = "./data/processed/labels.json"
    model_name = './output'
    max_length = 300
    # max_length = 500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载label
    labels_map = {}
    with open(labels_path, "r", encoding="utf-8") as r:
        labels = json.loads(r.read())
        for label in labels:
            label_id = labels[label]
            labels_map[label_id] = label

    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(labels_map))
    model.to(device)

    while True:
        text = input("请输入：")
        if not text or text == '':
            continue
        if text == 'q':
            break

        encoded_input = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
        input_ids = torch.tensor([encoded_input['input_ids']]).to(device)
        attention_mask = torch.tensor([encoded_input['attention_mask']]).to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        result = post_processing(outputs, text, labels_map)
        print(result)
        print(result['company'] if 'company' in result.keys() else '')


if __name__ == '__main__':
    main()

"""
   输入：根据北京市住房和城乡建设委员会总体工作部署，市建委调配给东城区118套房源，99户家庭全部来到现场
   识别结果：{'government': ['北京市住房和城乡建设委员会'], 'address': ['东城区']}
   
   输入：为星际争霸2冠军颁奖的嘉宾是来自上海新闻出版局副局长陈丽女士。最后，为魔兽争霸3项目冠军—
   识别结果：输入：为星际争霸2冠军颁奖的嘉宾是来自上海新闻出版局副局长陈丽女士。最后，为魔兽争霸3项目冠军—
   
   输入：作出对成钢违纪辞退处理决定，并开具了退工单。今年8月，公安机关以不应当追究刑事责任为由
   识别结果：{'name': ['成钢'], 'government': ['公安机关']}
"""






