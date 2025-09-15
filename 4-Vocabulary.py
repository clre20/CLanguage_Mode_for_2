# # pip install numpy
import ast
import numpy as np

data_i = "3-segmented_words.txt"
data_o1 = "4-word_to_id.txt"
data_o2 = "4-numerical_sequence.txt"

with open(data_i,'r',encoding='utf-8') as f:
    raw_string = f.read()

# 轉換列表
word_list = ast.literal_eval(raw_string)

# 建立一個空字典
word_to_id = {}
# 建立一個計數器
id_counter = 0

# 在詞彙表最前面加入 <unk> 和 <pad> 標記
# <unk> 代表未知詞
# <pad> 用於在序列長度不足時進行填充
word_to_id['<unk>'] = id_counter
id_counter += 1
word_to_id['<pad>'] = id_counter
id_counter += 1

for word in word_list:
    #檢查語詞是否已在字典裡
    if word not in word_to_id:
        # 如果不再字典裡，就加入字典，並給予編號
        word_to_id[word] = id_counter
        id_counter += 1

# 數字序列
numerical_sequence = []
for word in word_list:
    number = word_to_id.get(word, -1)  # 如果找不到就回傳 -1
    numerical_sequence.append(number)

with open(data_o1, 'w', encoding='utf-8') as f:
    f.write(str(word_to_id))

with open(data_o2, 'w', encoding='utf-8') as f:
    f.write(str(numerical_sequence))

print("完成編碼啦~")