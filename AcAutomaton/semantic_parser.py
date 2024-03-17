import jieba

from config import entity_corpus_path, entity_searcher_save_path, contexts
import ahocorasick
import pandas as pd
import os
import pickle


def build_search_tree(input_folder_path, tree_save_path):
    """读取股票名称，股东和概念实体，构建 ac 树"""
    # https://pypi.org/project/pyahocorasick/
    tree = ahocorasick.Automaton()

    stock_basic = pd.read_csv(os.path.join(input_folder_path, '股票信息.csv'), encoding='utf-8')

    # 遍历 stock_basic，添加 name 即股票名字
    # 股票名字为key，value表示为具体的实体类型，比如：tree.add_word('股票名A', ('股票名A', '股票'))
    for index, row in stock_basic.iterrows():
        tree.add_word(row['ts_code'], (row['ts_code'], '股票'))
        # tree.add_word(row['ts_code'], (index, row['ts_code']))

    concept = pd.read_csv(os.path.join(input_folder_path, '概念信息.csv'), encoding='utf-8')
    # 遍历 concept，添加 name 即概念名字
    # 概念名字为key，value表示为具体的实体类型，比如：tree.add_word('概念名A', ('概念名A', '概念'))
    for index, row in concept.iterrows():
        tree.add_word(row['name'], (row['name'], '概念'))
        # tree.add_word(row['name'], (index, row['name']))

    holder = pd.read_csv(os.path.join(input_folder_path, '股东信息.csv'), encoding='utf-8')
    # 遍历 holder，添加 股东名称
    # 股东名称为key，value表示为具体的实体类型，比如：tree.add_word('股东名称A', ('股东名称A', '股东'))
    for index, row in holder.iterrows():
        tree.add_word(row['股东名称'], (row['股东名称'], '股东'))
        # tree.add_word(row['股东名称'], (index, row['股东名称']))

    tree.make_automaton()

    with open(tree_save_path, 'wb') as fout:
        pickle.dump(tree, fout)


class SemanticParser:
    """实体搜索器"""

    def __init__(self, entity_model_load_path, question_types):
        self.entity_model_load_path = entity_model_load_path
        self.entity_model = self.load_model()
        self.question_types = question_types

    def load_model(self):
        """加载模型"""
        with open(self.entity_model_load_path, 'rb') as fin:
            return pickle.load(fin)

    def predict_question_types(self, query):
        """判断问题类型，这里只是通过关键词去判断，可以改成分类模型"""

        rtn_ques_types = []
        for ques_type, kws in self.question_types.items():
            for each_kw in kws:
                if each_kw in query:
                    rtn_ques_types.append(ques_type)
                    break
        return rtn_ques_types

    def predict(self, query):
        """预测 query"""

        rtn = {}

        # 预测类型
        ques_types = self.predict_question_types(query)

        # 预测实体
        entities = {}
        # # 如果query是中文文本，需要先进行分词
        # # seg_list = jieba.cut(query, cut_all=False)  # '14,000019.SZ的股东名称是什么？'
        # # seg_list = ['14,000019.SZ', '股东', '什么']  # '14,000019.SZ的股东名称是什么？'
        # # seg_list = ['000001.SZ', '股东', '陈建建']  # 000001.SZ的股东是陈建建吗？
        # seg_list = ['000301.SZ', '密集调研', '陈建建']  #
        # for word in seg_list:
        #     if word in self.entity_model:
        #         for end_index, (entity_name, entity_type) in self.entity_model.iter(word):
        #             entities[entity_name] = entity_type

        # 参考答案
        for end_index, (entity_name, entity_type) in self.entity_model.iter(query):
            entities[entity_name] = entity_type

        if len(ques_types) != 0 and len(entities) != 0:
            rtn['ques_types'] = ques_types
            rtn['entities'] = entities
            # 备份
            contexts['ques_types'] = ques_types
            contexts['entities'] = entities
        elif len(ques_types) != 0:
            rtn['ques_types'] = ques_types
            # 备份
            contexts['ques_types'] = ques_types

            # 从对话历史中继承问题类型
            rtn['entities'] = contexts['entities']
        elif len(entities) != 0:
            # 从对话历史中继承问题类型
            rtn['ques_types'] = contexts['ques_types']

            rtn['entities'] = entities
            # 备份
            contexts['entities'] = entities
        else:
            # 如果两个都没有找到，那说明是没有涉及 KG
            rtn['ques_types'] = []
            rtn['entities'] = {}

        return rtn


if __name__ == '__main__':

    print('开始训练实体搜索树...')

    build_search_tree(entity_corpus_path, entity_searcher_save_path)

    print('实体搜索树训练成功...')
