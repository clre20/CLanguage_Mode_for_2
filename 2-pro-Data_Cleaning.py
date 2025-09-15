# 2-Data_Cleaning_pro.py
# pip install beautifulsoup4
import os
from bs4 import BeautifulSoup

# 清潔函數
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    p_tag_list = soup.find_all("p")

    # 建立一個列表存放所有文字
    all_text = []

    for tag in p_tag_list:
        text = tag.get_text(strip=True)  # 去除前後空白
        if text:  # 避免空字串
            all_text.append(text)

    # 把所有文字連起來
    cleaned_text = "".join(all_text)
    return cleaned_text

# 資料夾與輸出檔
data_folder = "Data"
output_file = "2-cleaned_data.txt"

# 先清空輸出檔案
with open(output_file, 'w', encoding='utf-8') as f:
    pass

# 依序處理資料夾內的所有 html 檔
for filename in sorted(os.listdir(data_folder)):
    if filename.endswith(".html"):
        file_path = os.path.join(data_folder, filename)
        print(f"正在處理: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()

        # 呼叫清理函數
        cleaned_content = clean_html(html)

        # 將清理後內容寫入輸出檔，並加上結尾標記
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(cleaned_content + "\n[END_OF_NOVEL]\n")

print("所有資料清潔完成~")
