import re

text = """
这是一些普通文本。
### 意译
这是需要提取的文本内容，包括多行内容。
"""

match = re.search(r"### 意译\s*(.*)", text, re.DOTALL)
# r"### 意译\s*(.*)" 是一个原始字符串，其中包含了正则表达式模式。
# ### 意译：匹配文本中的“### 意译”字样。
# \s*：匹配“### 意译”后面可能存在的任意数量的空白字符（包括空格、制表符、换行符等）。
# (.*)：一个捕获组，用于匹配并捕获任意字符（包括换行符），直到文本末尾。捕获组会将匹配的内容保存下来供后续使用。
# re.DOTALL：正则表达式的标志，表示点号 . 还可以匹配换行符。这意味着 .* 可以匹配跨越多行的内容。

if match:
    print(match.group(1).strip())
    # if match: 检查 re.search 是否找到匹配内容。如果找到了，match 对象不为 None。
    # return match.group(1).strip()：group(1) 返回捕获组中匹配的文本，即“### 意译”之后的所有内容。strip() 方法移除匹配内容前后的任何空白字符（如空格和换行符），然后返回处理后的字符串。
else:
    print("No match found.")
