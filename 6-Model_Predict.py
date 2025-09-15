import torch
import torch.nn as nn
import ast
import jieba
import json

# 1. 載入模型架構（必須與訓練時相同）
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, vocab_size) # <-- 這裡

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        output = output.reshape(-1, self.lstm.hidden_size)
        final_output = self.fc1(output)
        return final_output

# 2. 載入詞彙表，並建立 word_to_id 和 id_to_word
with open("4-word_to_id.txt", 'r', encoding='utf-8') as f:
    word_to_id_str = f.read()
    word_to_id = ast.literal_eval(word_to_id_str)

id_to_word = {v: k for k, v in word_to_id.items()}

# 3. 定義模型參數，並載入權重
with open('config.json','r',encoding='utf-8')as f:
    config = json.load(f)
vocab_size = len(word_to_id)
embedding_dim = config['embedding_dim']
hidden_dim = config['hidden_dim']
seq_length = config['seq_length']

model = SimpleLSTM(vocab_size, embedding_dim, hidden_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.load_state_dict(torch.load("simple_lstm_model.pth"))
model.eval()

print("模型已成功載入！")

# 4. 預測函式
def predict_next_word(model, prompt_text, word_to_id, id_to_word, device, seq_length=3):
    # 將輸入的文字分詞
    words = list(jieba.cut(prompt_text))
    
    # 如果輸入的詞語數量少於序列長度，則進行填充
    if len(words) < seq_length:
        words = ['<unk>'] * (seq_length - len(words)) + words

    # 確保只取最後 seq_length 個詞語
    input_words = words[-seq_length:]

    # 將分詞後的詞語轉換成數字
    numerical_input = [word_to_id.get(word, word_to_id.get('<unk>')) for word in input_words]

    # 將數字序列轉換成張量
    input_tensor = torch.tensor([numerical_input]).to(device)

    # 得到模型預測結果
    with torch.no_grad():
        output = model(input_tensor)
    
    # 取得最終時間步的輸出
    final_output = output[-1, :]
    
    # 找出機率最高的索引
    predicted_id = torch.argmax(final_output, dim=-1).item()
    
    # 將預測結果轉換回文字
    predicted_word = id_to_word.get(predicted_id, '<unk>')
    
    return predicted_word

# 5. 在主程式中測試
# 你可以在這裡輸入你想讓模型預測的文字
prompt = "這裡放提示"
predicted_word = predict_next_word(model, prompt, word_to_id, id_to_word, device, seq_length)
print(f"輸入文字: {prompt}")

print(f"模型預測的下一個詞語是: {predicted_word}")
