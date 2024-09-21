import json
import numpy as np
import faiss
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 加载数据集
with open("../data/train-v2.0.json", "r") as f:
    squad_data = json.load(f)


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


# 使用 sentence-transformers库 获得句子的向量表示
def sentence_embedding(text):
    sent_embedding = model.encode(text)
    return sent_embedding


paragraph = [paragraph['context'] for article in squad_data['data'] for paragraph in article['paragraphs']]
# paragraph_vectors = [sentence_embedding(paragraph['context']) for article in squad_data['data'] for paragraph in article['paragraphs']]  # list:19035
# # 使用tqdm库显示进度条
# paragraph_vectors = [sentence_embedding(paragraph['context'])
#                      for article in tqdm(squad_data['data'], desc='processing articles')
#                      for paragraph in tqdm(article['paragraphs'], desc='processing paragraphs', leave=False)]  # leave=False 是为了避免内循环的进度条在完成后仍然留在屏幕上。
paragraph_vectors = [sentence_embedding(paragraph['context'])
                     for article in tqdm(squad_data['data'], desc='processing articles')  # miniters=1: 每处理一次循环，刷新一次进度条。mininterval=0.5: 每0.5秒更新一次进度。
                     for paragraph in article['paragraphs']]


# 构建索引
dimension = model.get_sentence_embedding_dimension()  # 384  # dimension 代表了每个段落向量的维度
faiss_index = faiss.IndexFlatL2(dimension)  # 构建索引，这里使用的是 faiss 库的 IndexFlatL2 索引，它是最简单的索引，只需要指定维度即可。  # dimension 指的是每个向量的维度。

# 将 NumPy 数组转换为 1 个二维数组
# paragraph_vectors 是一个列表，里面存储了各个段落的向量。np.stack() 将这些向量堆叠成一个二维 NumPy 数组，每行代表一个段落的向量。astype('float32') 将数组转换为 float32 类型。
paragraph_vectors = np.stack(paragraph_vectors).astype('float32')  # ndarray (19035, 110256)
faiss_index.add(paragraph_vectors)

# 存储FAISS索引到文件
faiss.write_index(faiss_index, "../data/faiss_index_squad.faiss")

# # 从文件中加载FAISS索引
# index = faiss.read_index("index.faiss")
#
# # # 如果你的索引是在 GPU 上构建的，在加载时需要将索引重新传回到 GPU # 将CPU索引转换为GPU索引
# # index_cpu = faiss.read_index("index.faiss")
# # res = faiss.StandardGpuResources()
# # index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)


# 相似性搜索
def search_for_paragraphs(search_term, num_results):
    search_vector = sentence_embedding(search_term)
    search_vector = np.array([search_vector]).astype('float32')
    distances, indexes = faiss_index.search(search_vector, num_results)
    for i, (distance, index) in enumerate(zip(distances[0], indexes[0])):
        print(f"Result {i + 1}, Distance: {distance}")
        print(paragraph[index])
        print()


search_term = "What is the capital of France?"
search_for_paragraphs(search_term, 5)
