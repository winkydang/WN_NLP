import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 知识语料路径
entity_corpus_path = '../data_1/knowledge/'

# 实体搜索器存储路径
entity_searcher_save_path = '../checkpoints/entity_searcher/search_tree.pkl'

# 实体搜索器加载路径
entity_searcher_load_path = './checkpoints/entity_searcher/search_tree.pkl'

# 分类器语料路径
classifier_corpus_path = '../data_1/classifier/chat.train'

# 分类器模型存储路径
classifier_save_path = '../checkpoints/classifier/model.bin'

# 分类器模型加载路径
classifier_load_path = './checkpoints/classifier/model.bin'

# 闲聊回复语料库
chat_responses = {
    'greet': [
        'hello，我是小A，小哥哥小姐姐有关于股票的问题可以问我哦',
        '你好，我是小A，输入股票名称或者代码查看详细信息哦',
        '你好，我是小A，可以问我股票相关的问题哦'
    ],
    'goodbye': [
        '再见',
        '不要走，继续聊会呗',
        '拜拜喽，别忘了给个小红心啊',
    ],
    'bot': [
        '没错，我就是集美貌与才智于一身的小A',
        '小A就是我，我就是小A'
    ],
    'safe': [
        '祝你吃嘛嘛香，身体倍棒',
        '人是铁饭是钢，一顿不吃得慌'
    ]
}

# 问题类型
question_types = {
    'concept':
        ['概念'],
    'holder':
        ['股东', '控制', '控股', '持有'],
    'industry':
        ['行业'],
}

# 存储对话历史中上一次涉及的问题类型和实体
contexts = {
    'ques_types': None,
    'entities': None
}

import os
from dotenv import load_dotenv, find_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 加载环境变量
load_dotenv(find_dotenv())

# 获取 OpenAI API 密钥
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# 获取 OpenAI API 服务地址
# OPENAI_API_BASE = os.environ['OPENAI_API_BASE'] # 使用国内API服务

# 官方文档 - Models：https://platform.openai.com/docs/models
MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4-turbo",
    "gpt-4",
    "gpt-3.5-turbo",
]
DEFAULT_MODEL = MODELS[0]

MODEL_TO_MAX_TOKENS = {
    "gpt-4o-mini": 16383,
    "gpt-4o": 4096,
    "gpt-4-turbo": 4096,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 4096,
}
DEFAULT_MODEL_MAX_TOKENS = MODEL_TO_MAX_TOKENS.get(DEFAULT_MODEL, 4096)


QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

DEFAULT_MAX_TOKENS = 2000




