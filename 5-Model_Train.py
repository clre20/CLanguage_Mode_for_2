# pip install torch numpy matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import ast
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime

# 2. 定義模型類別
# (我們之前討論的 SimpleLSTM 類別)
# 定義 LSTM 模型類別
class SimpleLSTM(nn.Module):
    #vocab_size (詞彙大小): 這就是我們在 4-Vocabulary.py 檔案裡建立的詞彙表的大小，也就是我們模型能夠辨識的不重複詞語總數。這決定了模型的輸入層有多「寬」。
    #embedding_dim (嵌入層維度): 這代表每個詞語的向量維度。你可以把它想像成一個詞語的「特徵」數量。如果 embedding_dim 是 100，那就表示每個詞語都會被轉換成一個包含 100 個數字的向量。
    #hidden_dim (隱藏層維度): 這代表 LSTM 層的「內部記憶體大小」。它決定了 LSTM 能夠儲存多少關於先前詞語的資訊。
    #output_dim (輸出層維度): 這代表模型的最終輸出，應該要等於我們的詞彙表總數（vocab_size）。
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        print("---LSTM---")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # 全連結層
        print("---全連結層---")
        self.fc1 = nn.Linear(hidden_dim, vocab_size) #1
        #self.fc3 = nn.Linear(hidden_dim, vocab_size) #3

    def forward(self, text):
        # 1. 詞嵌入層
        embedded = self.embedding(text)
        # 2. LSTM 層
        output, (hidden, cell) = self.lstm(embedded)
        # 3. 輸出層
        #output = output.reshape(-1, self.lstm.hidden_size)
        # 4.傳給第一層全連結層
        #output = F.relu(self.fc1(output))

        # 6.傳給第五層全連結層
        final_output = self.fc1(output)

        return final_output

# 3. 定義超參數
print("---載入超參數---")
with open('config.json','r',encoding='utf-8') as f:
    config = json.load(f)

vocab_size = 0  # 詞彙大小(會自動更新)
embedding_dim = config['embedding_dim'] # 嵌入層維度
hidden_dim = config['hidden_dim'] # 隱藏層維度
seq_length = config['seq_length']  # 序列長度
epochs = config['epochs'] # 訓練次數
learning_rate = config['learning_rate'] # 學習率

# 4. 讀取和準備資料
data_i_1 = "4-word_to_id.txt"
data_i_2 = "4-numerical_sequence.txt"

# 讀取詞彙表
print("---讀取詞彙表---")
with open(data_i_1, 'r', encoding='utf-8') as f:
    word_to_id_str = f.read()
    word_to_id = ast.literal_eval(word_to_id_str)

# 讀取數字序列
print("---讀取數字序列---")
with open(data_i_2, 'r', encoding='utf-8') as f:
    numerical_sequence_str = f.read()
    numerical_sequence = ast.literal_eval(numerical_sequence_str)

# 更新 vocab_size
print("---更新 vocab_size---")
vocab_size = len(word_to_id)
print(f"Token:{vocab_size}")

# 將數字序列切分成輸入和目標序列
print("---製作題目---")
input_sequences = []
target_sequences = []

for i in range(len(numerical_sequence) - seq_length):
    input_seq = numerical_sequence[i : i + seq_length]
    target_seq = numerical_sequence[i + 1 : i + 1 + seq_length]

    input_sequences.append(input_seq)
    target_sequences.append(target_seq)

# 將資料轉換為 PyTorch 張量
print("---將資料轉換為 PyTorch 張量---")
inputs = torch.tensor(input_sequences)
targets = torch.tensor(target_sequences)

# 4. 定義損失函式和優化器
print("---定義損失函式和優化器---")
model = SimpleLSTM(vocab_size, embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 4.1 檢查並設定設備
print("---檢查並設定設備---")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 裝置數量: {torch.cuda.device_count()}")
    print(f"CUDA 裝置名稱: {torch.cuda.get_device_name(0)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    print("未偵測到 CUDA 裝置。")

# 將模型移動到指定的設備上
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 5. 訓練迴圈
print("開始訓練")
loss_history = []  # 創建一個空的列表來記錄每一輪的損失值
accuracy_history = [] # 記錄準確率的列表
for epoch in range(epochs):
    model.train()
    
    # 將張量移動到指定的設備上
    inputs_to_device = inputs.to(device)
    targets_to_device = targets.to(device)
    
    optimizer.zero_grad()
    
    # 前向傳播
    outputs = model(inputs_to_device)
    
    # 計算損失
    loss = criterion(outputs.view(-1, outputs.size(-1)), targets_to_device.view(-1))

    # 計算準確率
    # 找出模型預測最有可能的詞語 (機率最高的索引)
    predictions = torch.argmax(outputs, dim=2)

    # 展平 predictions 和 targets 以進行比較
    predictions = predictions.view(-1)
    targets_flat = targets_to_device.view(-1)
    
    # 比較預測結果和正確答案
    correct_predictions = (predictions == targets_to_device.view(-1)).sum().item()
    
    # 計算當前的準確率
    total_predictions = targets_to_device.view(-1).size(0)
    accuracy = correct_predictions / total_predictions

    # 儲存損失
    loss_history.append(loss.item()) # 將這一次的損失值加進列表
    accuracy_history.append(accuracy) # 將準確率加入列表
    # 反向傳播
    loss.backward()
    
    # 更新權重
    optimizer.step()
    
    if (epoch + 1) % 1 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy}")

# 訓練完成，保存模型
torch.save(model.state_dict(), "simple_lstm_model.pth")
print("模型訓練完成~。")


# 取得當前日期和時間
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 準備參數字串
params_str = (
    f"詞彙大小(Token): {vocab_size}\n"
    f"嵌入維度: {config['embedding_dim']}\n"
    f"隱藏維度: {config['hidden_dim']}\n"
    f"序列長度: {config['seq_length']}\n"
    f"訓練次數: {config['epochs']}\n"
    f"學習率: {config['learning_rate']}"
)

# 尋找並設定一個中文字體
font_path = "C:/Windows/Fonts/msjh.ttc"
font_prop = FontProperties(fname=font_path, size=10)

# 設定 Matplotlib 的中文字體
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

# 繪製損失曲線圖
fig, ax1 = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(right=0.80)

# 上標題
plt.title('訓練過程中的損失與準確率(Training Loss and Accuracy over Epochs)', fontproperties=font_prop)
# 圖表下方添加參數資訊
plt.figtext(0.865, 0.87, params_str, ha="left", va="top", fontsize=10, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
# X標籤
plt.xlabel(f'訓練次數(Epochs)\n 時間 ({current_time}) By CLRE-20')
# Y標籤
#plt.ylabel('Loss', fontproperties=font_prop)

# 繪製線條
# === 左邊的 Y 軸 (損失) ===
ax1.set_ylabel('損失 (Loss)', color='tab:blue', fontproperties=font_prop)
ax1.plot(loss_history, color='tab:blue', label='訓練損失')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_ylim(0, 10)
ax1.set_yticklabels(ax1.get_yticklabels(), fontproperties=font_prop)  # 刻度文字使用字體

# === 右邊的 Y 軸 (準確率) ===
ax2 = ax1.twinx()  # 創建一個共享 X 軸的第二個 Y 軸
ax2.set_ylabel('準確率 (Accuracy)', color='tab:orange', fontproperties=font_prop)
ax2.plot(accuracy_history, color='tab:orange', label='準確率')
ax2.tick_params(axis='y', labelcolor='tab:orange')
ax2.set_ylim(0, 1) # 設定準確率的 Y 軸範圍在 0 到 1 之間
ax2.set_yticklabels(ax2.get_yticklabels(), fontproperties=font_prop)  # 刻度文字使用字體

# 在最後一個數據點標註損失值
last_epoch_index = len(loss_history) - 1
last_loss = loss_history[-1]
ax1.text(last_epoch_index, last_loss, f'{last_loss:.4f}', ha='center', va='top', fontproperties=font_prop)
# 在最後一個數據點標註準確率
last_accuracy = accuracy_history[-1]
ax2.text(last_epoch_index, last_accuracy, f'{last_accuracy:.4f}', ha='center', va='top', fontproperties=font_prop)

# 繪製圖例和網格線
ax1.legend(loc='upper left', prop=font_prop)  # 指定圖例字體
ax2.legend(loc='upper right', prop=font_prop)  # 指定圖例字體
plt.grid(True)
# 檔名
plt.savefig(f'training_loss_plot_{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}.png')
# 顯示圖表
plt.show()
print("訓練損失曲線圖已經畫完囉~")