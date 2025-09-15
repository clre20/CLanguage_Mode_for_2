# pip install jieba
import jieba

# 手動把特殊標記加入詞典，確保不會被拆開
jieba.add_word("[END_OF_NOVEL]")

# 分詞函數
def segment_text(text):
    seg_list = jieba.cut(text, cut_all=False)
    words = list(seg_list)
    # 過濾空白字元
    cleaned_words = [word for word in words if word.strip() != '']
    return cleaned_words

data_i = "2-cleaned_data.txt"
data_o = "3-segmented_words.txt"

with open(data_i, 'r', encoding='utf-8') as f:
    text = f.read()

# 呼叫分詞函式
segmented_words = segment_text(text)

# 輸出保持 list 格式（例如 ["A","B","C"]）
final_output = str(segmented_words)

with open(data_o, 'w', encoding='utf-8') as f:
    f.write(final_output)

print("完成分詞啦~")
