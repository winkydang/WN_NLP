import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties

# 设置matplotlib支持中文字体的方式
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种支持中文的字体 # 这种适用于windows系统中
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 'Arial Unicode MS' 是一种支持中文的字体 # 这种适用于MAC系统中
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 创建一个有向图
G = nx.DiGraph()

# 添加带标签的节点
G.add_node("中国核工业集团有限公司", pos=(0, 1))
G.add_node("新华水利控股集团有限公司", pos=(2, 1))
G.add_node("新华水力发电有限公司", pos=(1, 0))

# 添加带权重（持股比例）的边
G.add_edge("中国核工业集团有限公司", "新华水力发电有限公司", weight=57.65)
G.add_edge("新华水利控股集团有限公司", "新华水力发电有限公司", weight=42.35)

# 定义节点的位置
pos = nx.get_node_attributes(G, 'pos')

# 调整图形尺寸
plt.figure(figsize=(8, 6))

# 画节点
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=8000)

# 画边
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20, edge_color='black')

# 画节点的标签
nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', verticalalignment='center', horizontalalignment='center')

# 画边的标签
edge_labels = {('中国核工业集团有限公司', '新华水力发电有限公司'): '控股 57.65%',
               ('新华水利控股集团有限公司', '新华水力发电有限公司'): '控股 42.35%'}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

# 隐藏坐标轴
plt.axis('off')

# 显示标题
plt.title("股权结构图")

# 显示图表
plt.show()
