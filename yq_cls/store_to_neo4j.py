import os

from tqdm import tqdm
import pandas as pd
from py2neo import Graph, Node, Relationship, NodeMatcher

from config import BASE_DIR

# --------------------------- 连接 Neo4j
# 官方文档：https://py2neo.org/2021.1/
# graph = Graph('http://localhost:7474/finance_demo/db/', auth=('neo4j', 'neo4j123'))
graph = Graph('bolt://localhost:7687/finance_demo/yq/', auth=('neo4j', 'yqneo4j123'))
print(graph)

# --------------------------- 创建实体
# 公司
print('创建 公司 实体...')
company_basic = pd.read_csv(os.path.join(BASE_DIR, 'yq_cls/data_1/yq_1k.csv'), encoding='utf-8')
for idx, each_row in tqdm(company_basic.iterrows()):
    # 方法说明：https://py2neo.org/2021.1/data/index.html#py2neo.data.Node
    # 公司 是 label
    # keyword arguments 是属性，如 公司名称、公司类型 等
    each_company = Node('公司',
                      公司名称=each_row['COMPANYNAME'],
                      公司类型=each_row['COMPANYFLAG'])
    try:
        # 方法说明：https://py2neo.org/2021.1/workflow.html#py2neo.Transaction.create
        graph.create(each_company)
    except Exception as e:
        print(f'Error: {e}, data_1 idx: {idx}, data_1: {each_row}')

# 概念
print('创建 新闻 实体...')
news = pd.read_csv(os.path.join(BASE_DIR, 'yq_cls/data_1/yq_1k.csv'), encoding='utf-8')
for idx, each_row in tqdm(news.iterrows()):
    each_news = Node('新闻',
                        新闻标题=each_row['TITLE'],
                        新闻内容=each_row['CONTENT'],
                        新闻类型=each_row['TYPE'],
                        新闻来源=each_row['DATASOURCE'],
                        链接=each_row['LINKURL'],
                        发布时间=each_row['PUBLISHDATE'])
    graph.create(each_news)

# --------------------------- 创建关系
# 方法说明：https://py2neo.org/2021.1/matching.html#py2neo.NodeMatcher
matcher = NodeMatcher(graph)

# 股票-概念
print('创建 公司-新闻 关系...')
company_news = pd.read_csv(os.path.join(BASE_DIR, 'yq_cls/data_1/yq_1k.csv'), encoding='utf-8')
for idx, each_row in tqdm(company_news.iterrows()):
    # first() 方法返回第一个匹配的 Node，如果找不到则返回 None
    node1 = matcher.match('公司', 公司名称=each_row['COMPANYNAME']).first()
    node2 = matcher.match('新闻', 新闻标题=each_row['TITLE']).first()
    if node1 is not None and node2 is not None:
        # 方法说明：https://py2neo.org/2021.1/data/index.html#py2neo.data.Relationship
        # 格式：Relationship(start_node, type, end_node)
        r = Relationship(node1, '相关新闻', node2,
                         一级分类=each_row['CLASSIFICATION_1'],
                         二级分类=each_row['CLASSIFICATION_2'],
                         三级分类=each_row['CLASSIFICATION_3'])
        graph.create(r)


print('公司-新闻 关系 导入成功...')
