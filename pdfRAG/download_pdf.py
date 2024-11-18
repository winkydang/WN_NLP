# 从excel读取pdf链接，批量下载pdf文件
import os

import pandas as pd

from pdfRAG.utils.utils import download_pdf
from settings import BASE_DIR

# 自定义表头
custom_headers = ['COMPANYNAME', 'TITLE', 'CONTENT', 'EMORATE', 'IMPORTANCE', 'CLASSIFICATION_1', 'CLASSIFICATION_2', 'CLASSIFICATION_3', 'TYPE', 'DATASOURCE', 'LINKURL', 'PUBLISHDATE', 'COMPANYFLAG']
# df1 = pd.read_csv('./data/pdf公告_10条.csv', names=custom_headers)
df1 = pd.read_csv('./data/pdf公告_2319条_241012.csv', names=custom_headers)
urls = df1['LINKURL'].values.tolist()

# # 读取csv文件，指定不使用表头
# df1 = pd.read_csv('./data/pdf公告_2319条_241012.csv', header=None)
# urls = df1[10].values.tolist()

save_path = os.path.join(BASE_DIR, 'pdfRAG/data/pdfs')

# 使用 itertuples() 逐行遍历 DataFrame
for row in df1.itertuples():
    url = row.LINKURL
    download_pdf(url, save_path + '/' + row.TITLE + '.pdf')









