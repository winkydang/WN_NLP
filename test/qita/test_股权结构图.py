import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

# 设置matplotlib支持中文字体的方式
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种支持中文的字体 # 这种适用于windows系统中
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 'Arial Unicode MS' 是一种支持中文的字体 # 这种适用于MAC系统中
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# Create a directed graph
G = nx.DiGraph()

# Add nodes with labels
G.add_node("中国核工业集团有限公司", pos=(0, 1))
G.add_node("新华水利控股集团有限公司", pos=(2, 1))
G.add_node("新华水力发电有限公司", pos=(1, 0))

# Add edges with labels (ownership percentages)
G.add_edge("中国核工业集团有限公司", "新华水力发电有限公司", weight=57.65)
G.add_edge("新华水利控股集团有限公司", "新华水力发电有限公司", weight=42.35)

# Define positions of nodes
pos = nx.get_node_attributes(G, 'pos')

# Draw the nodes
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000)

# Draw the edges
nx.draw_networkx_edges(G, pos, edgelist=G.edges(), arrowstyle='-|>', arrowsize=20, edge_color='black')

# Draw the labels
nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

# Draw edge labels
edge_labels = {('中国核工业集团有限公司', '新华水力发电有限公司'): '控股 57.65%',
               ('新华水利控股集团有限公司', '新华水力发电有限公司'): '控股 42.35%'}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

# Show plot
plt.title("股权结构图")
plt.axis('off')
plt.show()
