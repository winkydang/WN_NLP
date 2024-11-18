"""
如果你想将读取的CSV数据保存为PDF，可以使用 matplotlib 或 reportlab 等库将表格数据渲染成PDF文件。
"""
import pandas as pd
import matplotlib.pyplot as plt
# 设置matplotlib支持中文字体的方式
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是一种支持中文的字体 # 这种适用于windows系统中
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 'Arial Unicode MS' 是一种支持中文的字体 # 这种适用于MAC系统中
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取CSV数据，不设置表头或自定义表头
df = pd.read_csv('../data/pdf公告_10条.csv', header=None)  # or use `names=custom_headers`

# 创建一个新图像
fig, ax = plt.subplots(figsize=(8, 6))

# 隐藏坐标轴
ax.axis('tight')
ax.axis('off')

# 使用pandas的dataframe绘制表格
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# 保存为PDF
plt.savefig('./data/output.pdf', bbox_inches='tight')

print("PDF文件已保存到: ./data/output.pdf")
