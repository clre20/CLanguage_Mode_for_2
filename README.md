<h1 align="center">CLanguage_Mode_for_2</h1>
<p align="center">CLRE-20</p>
<p align="center"><a href="/README.md">中文版</a> | <a href="/EREADME.md">英文版</a></p>
---
## README.zh.md (中文)

# 基於 LSTM 的中文文字生成模型

本專案展示了一個以 **PyTorch** 實作的 **LSTM（長短期記憶）神經網路文字生成模型**。
流程涵蓋 **資料收集 → 清理 → 分詞 → 詞彙表建立 → 模型訓練 → 文字生成**，適合作為中文語言模型的基礎實作範例。

## 功能特色

* **資料清理**：支援從 HTML `<p>` 標籤與 PDF 頁面範圍抽取文字。
* **中文分詞**：使用 `jieba` 分詞，支援自訂特殊標記（如 `[END_OF_NOVEL]`）。
* **詞彙表建立**：將詞彙轉換為數值 ID，並提供 `<unk>` 與 `<pad>` 等特殊符號。
* **模型訓練**：提供 `SimpleLSTM` 訓練腳本，使用交叉熵損失與 Adam 優化器，並繪製 Loss/Accuracy 曲線。
* **文字預測**：可載入已訓練模型，依據提示詞產生後續文字。
* **參數設定**：所有主要超參數集中於 `config.json`，方便調整。

## 專案結構

| 檔案 / 資料夾                     | 說明                   |
| ---------------------------- | -------------------- |
| `1-DLweb.py`                 | 下載單一網頁 HTML          |
| `2-Data_Cleaning.py`         | 抽取單一 HTML `<p>` 文字   |
| `2-pro-Data_Cleaning.py`     | 批次處理多個 HTML 檔案       |
| `2-PDF.py`                   | 從 PDF 指定頁面抽取文字       |
| `3-Word_Segmentation.py`     | 基本中文分詞               |
| `3-pro-Word_Segmentation.py` | 強化版：確保特殊標記不被拆分       |
| `4-Vocabulary.py`            | 建立詞彙表與數值化資料          |
| `5-Model_Train.py`           | 訓練 LSTM 模型並繪製訓練曲線    |
| `6-Model_Predict.py`         | 使用已訓練模型進行文字預測        |
| `config.json`                | 超參數設定檔               |
| `Data/`                      | 放置原始資料（HTML、PDF、文字檔） |
| `*.txt / *.png / *.pth`      | 訓練過程產生的檔案            |

## 系統需求

請先安裝必要套件：

```bash
pip install torch numpy matplotlib beautifulsoup4 pypdf requests jieba
```

## 使用方法

1. **資料準備**

   * 單一 HTML → `1-DLweb.py` → `2-Data_Cleaning.py`
   * 多個 HTML → `2-pro-Data_Cleaning.py`
   * PDF → 修改 `2-PDF.py` 參數並執行

2. **中文分詞**

   * 基本分詞：`3-Word_Segmentation.py`
   * 加入特殊標記：`3-pro-Word_Segmentation.py`

3. **詞彙表與數值化**

   ```bash
   python 4-Vocabulary.py
   ```

4. **模型訓練**

   ```bash
   python 5-Model_Train.py
   ```

   輸出：

   * `simple_lstm_model.pth` → 模型權重
   * `training_loss_plot_*.png` → 訓練曲線

5. **文字生成**
   修改 `6-Model_Predict.py` 中的 `prompt`，再執行：

   ```bash
   python 6-Model_Predict.py
   ```
