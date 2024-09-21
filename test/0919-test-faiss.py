import json
import numpy as np
import faiss
import json
import tqdm
from sentence_transformers import SentenceTransformer
# import nltk
# from nltk.tokenize import word_tokenize
# import faulthandler
# faulthandler.enable()


# nltk.download('punkt_tab')

# 加载数据集
with open("../data/train-v2.0.json", "r") as f:
    squad_data = json.load(f)

# # 预处理数据  单词去重  得到词表 并索引化
# vocabulary = set(word for article in squad_data['data'] for paragraph in article['paragraphs'] for word in word_tokenize(paragraph['context']))
# word_to_index = {word: index for index, word in enumerate(vocabulary)}
#
#
# def convert_text_to_vector(text):  # 将文本转化为向量。使用 one-hot ，01编码
#     words = word_tokenize(text)
#     bow_vector = np.zeros(len(vocabulary))
#     for word in words:
#         if word in word_to_index:
#             bow_vector[word_to_index[word]] = 1
#     return bow_vector


# 使用 sentence-transformers库 获得句子的向量表示
def sentence_embedding(text):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    sent_embedding = model.encode([text])
    return sent_embedding[0]


paragraph = [paragraph['context'] for article in squad_data['data'] for paragraph in article['paragraphs']]
# paragraph_vectors = [convert_text_to_vector(paragraph['context']) for article in squad_data['data'] for paragraph in article['paragraphs']]  # list:19035
paragraph_vectors = [sentence_embedding(paragraph['context']) for article in squad_data['data'] for paragraph in article['paragraphs']]  # list:19035
# 构建索引
# dimension = len(vocabulary)  # 110256
dimension = len(paragraph)  # 19035
faiss_index = faiss.IndexFlatL2(dimension)
# 将 NumPy 数组转换为 1 个二维数组
paragraph_vectors = np.stack(paragraph_vectors).astype('float32')  # ndarray (19035, 110256)
faiss_index.add(paragraph_vectors)


# 相似性搜索
def search_for_paragraphs(search_term, num_results):
    # search_vector = convert_text_to_vector(search_term)
    search_vector = sentence_embedding(search_term)
    search_vector = np.array([search_vector]).astype('float32')
    distances, indexes = faiss_index.search(search_vector, num_results)
    for i, (distance, index) in enumerate(zip(distances[0], indexes[0])):
        print(f"Result {i + 1}, Distance: {distance}")
        # print(squad_data['data'][index]['paragraphs'][0]['context'])
        print(paragraph[index])
        print()


search_term = "What is the capital of France?"
search_for_paragraphs(search_term, 5)
