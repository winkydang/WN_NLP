import ahocorasick
import jieba

# 初始化Aho-Corasick自动机
A = ahocorasick.Automaton()

# 假设我们有下列股票名称列表
stocks = ['平安银行', '万科A', '贵州茅台']

# 将股票名称添加到自动机中，并为每个股票名称分配一个唯一的ID
for idx, stock in enumerate(stocks):
    A.add_word(stock, (idx, stock))

# 构建自动机
A.make_automaton()

# 某段可能包含股票名称的文本
text = "张三在研究贵州茅台和平安银行的股票报告，他对万科A也很感兴趣。"

# 使用jieba进行中文分词
# seg_list = jieba.cut(text, cut_all=False)
seg_list = ['张三在', '研究', '贵州茅台', '股票', '平安银行', '万科A', '兴趣']

# 存储识别出的股票名称
found_stocks = set()

# 遍历分词结果，使用自动机检查每个词是否为股票名称
for word in seg_list:
    if word in A:
        for _, (idx, stock) in A.iter(word):
            found_stocks.add(stock)

# 输出识别出的股票名称
print("文本中识别出的股票名称：", found_stocks)
# 文本中识别出的股票名称： {'贵州茅台', '平安银行', '万科A'}
