import os.path

import requests

from settings import BASE_DIR


def download_pdf(url, save_path):
    try:
        # 发送请求获取PDF文件
        response = requests.get(url)

        # 检查请求是否成功
        if response.status_code == 200:
            # 将内容写入本地文件
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"PDF文件成功下载并保存到: {save_path}")
        else:
            print(f"下载失败，状态码: {response.status_code}")
    except Exception as e:
        print(f"下载失败，错误信息: {e}, url: {url}, save_path: {save_path}")
