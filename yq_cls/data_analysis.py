# 数据分析
# 分析新闻文本长度，分析分句后的每句的长度
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from config import BASE_DIR

# 设置显示风格
plt.style.use('fivethirtyeight')
# 设置matplotlib支持中文字体的方式
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种支持中文的字体 # 这种适用于windows系统中
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 'Arial Unicode MS' 是一种支持中文的字体 # 这种适用于MAC系统中

font = {'family': 'Arial Unicode MS', 'size': 12}
plt.rc('font', **font)
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# # 旋转x轴标签
# plt.xticks(rotation=90)

# 或者增加图表大小
plt.figure(figsize=(10, 8))  # 设置图表的尺寸，plt.figure(figsize=(宽度, 高度))

# # 调整字体大小
# plt.xticks(rotation=90, fontsize=8)


# # 一、文本长度分布
# # 绘制文本总长度分布图；文本分句后句子个数分布图；分句后子句的长度分布图
# def plot_length(path, title):
#     df = pd.read_csv(path)
#     # 在数据中添加句子长度列
#     df["len_content"] = list(map(lambda x: len(str(x)), df["CONTENT"]))
#     df["num_sentence"] = list(map(lambda x: len(eval(x)), df["CONTENT_CLEANED"]))
#     # df["num_sentence"] = list(df["CONTENT_CLEANED"]).apply(len)
#     # print(df.head())
#
#     print(title + "绘制文本总长度分布图：")
#     sns.countplot(x="len_content", data=df)  # 指定x轴上显示的数据列是DataFrame df中的"len_content"列
#     # 主要关注count(直方图)的纵坐标，横坐标通过下面的dist(密度图)来看
#     plt.xticks([])
#     # 设置x轴和y轴的名称
#     plt.xlabel('直方图-文本总长度')
#     plt.ylabel('直方图-文本总长度计数')
#     plt.savefig(os.path.join(BASE_DIR, 'yq_cls/tmp/data/res_data_analysis/直方图-文本总长度.png'))
#     plt.show()
#
#     sns.displot(df["len_content"])
#     # # 主要关注dist的横坐标，不需要绘制纵坐标
#     # plt.yticks([])
#     # 设置x轴和y轴的名称
#     plt.xlabel('dist-文本总长度')
#     plt.ylabel('dist-文本总长度计数')
#     plt.savefig(os.path.join(BASE_DIR, 'yq_cls/tmp/data/res_data_analysis/密度图-文本总长度.png'))
#     plt.show()
#
#     print(title + "文本分句后子句个数的分布图：")
#     sns.countplot(x="num_sentence", data=df)
#     plt.xticks([])  # 主要关注count(直方图)的纵坐标，横坐标通过下面的dist(密度图)来看
#     # 设置x轴和y轴的名称
#     plt.xlabel('直方图-子句个数')
#     plt.ylabel('直方图-子句个数计数')
#     plt.xticks(fontsize=4)  # 将字体大小设置为5 # 减小横轴标签的字体大小也可以帮助缓解重叠的问题。
#     plt.savefig(os.path.join(BASE_DIR, 'yq_cls/tmp/data/res_data_analysis/直方图-文本子句个数.png'))
#     plt.show()
#
#     sns.displot(df["num_sentence"])
#     # plt.yticks([])  # 主要关注dist的横坐标，不需要绘制纵坐标
#     # 设置x轴和y轴的名称
#     plt.xlabel('dist-子句个数')
#     plt.ylabel('dist-子句个数计数')
#     plt.savefig(os.path.join(BASE_DIR, 'yq_cls/tmp/data/res_data_analysis/密度图-文本子句个数.png'))
#     plt.show()
#
#     df.to_csv(path, index=False)  # 将添加了新列的df保存下来
#
#
# # 调用绘图函数
# title = "新闻文本分析"
# file_path = os.path.join(BASE_DIR, 'yq_cls/tmp/data/yq_clear_50w.csv')
# plot_length(file_path, title)


# # 二、对文本长度进行限制过滤
# def length_limit(path, length):
#     """
#     对文本长度进行限制过滤
#     :param path: ⻓度限制过滤前的⽂件路径，同时也是⻓度限制后的⽂件路径（覆盖）
#     :param length: ⻓度限制过滤的值
#     """
#     df = pd.read_csv(path)
#     print("过滤前的数量", len(df))
#     # 过滤掉长度超过限制的文本
#     df = df.drop(df[df["len_content"] > length].index)
#
#     # 删除长度列，只留下需要的列
#     df = df.drop(['len_content', 'num_sentence'], axis=1)
#     df.to_csv(path, index=True)
#     print("过滤后的数量", len(df))
#
#
# # 调用过滤长度函数
# file_path = os.path.join(BASE_DIR, 'yq_cls/tmp/data/yq_clear_50w.csv')
# length_limit(file_path, 1000)


# 三、标签数量分布
def plot_lable_count(file_path, save_path):
    df = pd.read_csv(file_path)
    # 使用Seaborn的countplot函数绘制柱状图
    sns.countplot(x="EMORATE", data=df)

    # 设置图形标题
    plt.title("样本标签数量分布图")

    plt.ylabel('标签')
    plt.ylabel('标签数量')
    # 添加表格线
    plt.grid(True)
    plt.savefig(save_path)
    plt.show()


# 调用  绘制样本标签数量分布
file_path = os.path.join(BASE_DIR, 'yq_cls/tmp/data/yq_clear_50w.csv')
save_path = os.path.join(BASE_DIR, 'yq_cls/tmp/data/res_data_analysis/柱状图-样本标签数量.png')
plot_lable_count(file_path, save_path)
