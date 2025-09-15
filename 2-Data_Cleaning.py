# pip install beautifulsoup4
from bs4 import BeautifulSoup

# 清潔函數
def clean_html(html):
    soup = BeautifulSoup(html, "html.parser")
    p_tage_list = soup.find_all("p")

    # 建立一個列表存放所有文字
    all_text = []

    for tag in p_tage_list:
        text = tag.text
        all_text.append(text)

    # 把所有文字連起來
    cleaned_text = "".join(all_text)

    return cleaned_text

data_i = "1-web_data.txt"
data_o = "2-cleaned_data.txt"

with open(data_i,'r',encoding='utf-8') as f:
    html = f.read()

# 呼叫清理函數
cleaned_content = clean_html(html)

with open(data_o,'w',encoding='utf-8') as f:
    f.write(cleaned_content)

print("資料清潔完畢~")