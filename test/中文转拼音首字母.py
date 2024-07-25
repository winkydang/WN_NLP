import os
import pandas as pd
from pypinyin import pinyin, Style
from config import BASE_DIR


# 定义一个函数来转换短语到拼音首字母
def to_initials(phrase):
    return ''.join([p[0][0] for p in pinyin(phrase, style=Style.FIRST_LETTER)])


df = pd.read_excel(os.path.join(BASE_DIR, 'data_1/组合.xlsx'))
# 应用这个函数到DataFrame的每个短语
df['PY_VC_FUND_NAME'] = df['VC_FUND_NAME'].apply(to_initials)
df['PY_VC_FUND_CAPTION'] = df['VC_FUND_CAPTION'].apply(to_initials)

# # 选择要保存的列
df_selected = df[['VC_FUND_CODE', 'VC_FUND_NAME', 'PY_VC_FUND_NAME', 'VC_FUND_CAPTION', 'PY_VC_FUND_CAPTION']]
# 保存修改后的DataFrame回CSV
save_path = os.path.join(BASE_DIR, 'data_1/组合_拼音.xlsx')
df_selected.to_excel(save_path, index=False)  # index=False表示在保存时不包括行索引




