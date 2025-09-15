# 將網頁下載
# pip install requests
import requests

#下載網頁函數
def DL_web(url):
    response = requests.get(url)

    # 網頁正常訪問?
    if response.status_code == 200:
        return response.text
    else:
        print(f"狀態碼為{response.status_code}")
        return None
    

url = "https://futurecity.cw.com.tw/article/3265"
data_o = "1-web_data.txt"

html = DL_web(url)

with open(data_o,'w',encoding='utf-8') as f:
    f.write(html)

print("資料儲存完畢~")