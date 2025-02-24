import os

from dotenv import load_dotenv, find_dotenv
from kg_gen import KGGen

load_dotenv(find_dotenv())

# 设置你的 OpenAI API 密钥
key = os.getenv("OPENAI_API_KEY")
# openai = OpenAI()

# Initialize KGGen with optional configuration
kg = KGGen(
    model="gpt-4o-mini",  # Default model
    temperature=0.0,  # Default temperature
    api_key= key # Optional if set in environment
)

# EXAMPLE 1: Single string with context
text_input = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
graph_1 = kg.generate(
    input_data=text_input,
    context="Family relationships"
)
# Output:
# entities={'Linda', 'Ben', 'Andrew', 'Josh'}
# edges={'is brother of', 'is father of', 'is mother of'}
# relations={('Ben', 'is brother of', 'Josh'),
#           ('Andrew', 'is father of', 'Josh'),
#           ('Linda', 'is mother of', 'Josh')}

# EXAMPLE 2: Large text with chunking and clustering
with open('../data/large_text.txt', 'r') as f:
    large_text = f.read()

# Example input text:
# """
# Neural networks are a type of machine learning model. Deep learning is a subset of machine learning
# that uses multiple layers of neural networks. Supervised learning requires training data to learn
# patterns. Machine learning is a type of AI technology that enables computers to learn from data.
# AI, also known as artificial intelligence, is related to the broader field of artificial intelligence.
# Neural nets (NN) are commonly used in ML applications. Machine learning (ML) has revolutionized
# many fields of study.
# ...
# """

graph_2 = kg.generate(
    input_data=large_text,
    chunk_size=5000,  # Process text in chunks of 5000 chars
    cluster=True  # Cluster similar entities and relations
)
# Output:
# entities={'neural networks', 'deep learning', 'machine learning', 'AI', 'artificial intelligence',
#          'supervised learning', 'unsupervised learning', 'training data', ...}
# edges={'is type of', 'requires', 'is subset of', 'uses', 'is related to', ...}
# relations={('neural networks', 'is type of', 'machine learning'),
#           ('deep learning', 'is subset of', 'machine learning'),
#           ('supervised learning', 'requires', 'training data'),
#           ('machine learning', 'is type of', 'AI'),
#           ('AI', 'is related to', 'artificial intelligence'), ...}
# entity_clusters={
#   'artificial intelligence': {'AI', 'artificial intelligence'},
#   'machine learning': {'machine learning', 'ML'},
#   'neural networks': {'neural networks', 'neural nets', 'NN'}
#   ...
# }
# edge_clusters={
#   'is type of': {'is type of', 'is a type of', 'is a kind of'},
#   'is related to': {'is related to', 'is connected to', 'is associated with'
#  ...}
# }

# EXAMPLE 3: Messages array
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
]
graph_3 = kg.generate(input_data=messages)
# Output:
# entities={'Paris', 'France'}
# edges={'has capital'}
# relations={('France', 'has capital', 'Paris')}

# EXAMPLE 4: Combining multiple graphs
text1 = "Linda is Joe's mother. Ben is Joe's brother."

# Input text 2: also goes by Joe."
text2 = "Andrew is Joseph's father. Judy is Andrew's sister. Joseph also goes by Joe."

graph4_a = kg.generate(input_data=text1)
graph4_b = kg.generate(input_data=text2)

# Combine the graphs
combined_graph = kg.aggregate([graph4_a, graph4_b])

# # Optionally cluster the combined graph
# clustered_graph = kg.cluster(
#     combined_graph,
#     context="Family relationships"
# )

print(graph_1)
# print(graph_2)
# print(graph_3)

# Output:
# entities={'Linda', 'Ben', 'Andrew', 'Joe', 'Joseph', 'Judy'}
# edges={'is mother of', 'is father of', 'is brother of', 'is sister of'}
# relations={('Linda', 'is mother of', 'Joe'),
#           ('Ben', 'is brother of', 'Joe'),
#           ('Andrew', 'is father of', 'Joe'),
#           ('Judy', 'is sister of', 'Andrew')}
# entity_clusters={
#   'Joe': {'Joe', 'Joseph'},
#   ...
# }
# edge_clusters={ ... }