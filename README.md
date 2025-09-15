# 基於 LSTM 的中文文字生成模型

這個專案包含一個使用 PyTorch 函式庫建構的 LSTM（長短期記憶）神經網路文字生成模型。整個流程涵蓋了從資料準備到模型訓練及文字預測的所有階段，旨在展示如何建構一個基礎的中文語言模型。

## 功能特色

  * **資料清理**：提供腳本從 HTML 的 `<p>` 標籤中提取文字，或從指定頁面的 PDF 檔案中提取文字。
  * **中文分詞**：使用 `jieba` 函式庫進行中文分詞，並能處理如 `[END_OF_NOVEL]` 等特殊標記。
  * **詞彙表建立**：將分詞後的文字列表轉換為數值化的詞彙表，為每個詞語指定唯一的 ID。同時也包含用於未知詞 (`<unk>`) 和填充 (`<pad>`) 的特殊標記。
  * **模型訓練**：定義並訓練一個 `SimpleLSTM` 模型，使用交叉熵損失函數和 Adam 優化器。訓練過程會產生圖表，視覺化損失和準確率隨訓練次數的變化。
  * **文字預測**：提供一個腳本，用於載入已訓練的模型，並根據給定的提示詞來預測下一個詞語。
  * **參數設定**：所有主要超參數都集中在 `config.json` 檔案中，方便進行調整。

## 專案結構

  * `1-DLweb.py`：用於下載網頁 HTML 內容的腳本。
  * `2-Data_Cleaning.py`：從單一 HTML 檔案中提取 `<p>` 標籤內文字的清理腳本。
  * `2-pro-Data_Cleaning.py`：強化版，可處理一個資料夾內的多個 HTML 檔案。
  * `2-PDF.py`：從 PDF 檔案的指定頁面範圍提取文字的腳本。
  * `3-Word_Segmentation.py`：執行基礎中文分詞的腳本。
  * `3-pro-Word_Segmentation.py`：進階版，將特殊標記手動加入 `jieba` 詞典，確保其不被拆分。
  * `4-Vocabulary.py`：建立詞語與 ID 的對應表及數值化序列的腳本。
  * `5-Model_Train.py`：主要的訓練腳本，定義模型、載入資料並進行訓練，同時繪製訓練曲線圖。
  * `6-Model_Predict.py`：用於載入已訓練模型並進行文字預測的腳本。
  * `config.json`：存放模型超參數的設定檔。
  * `simple_lstm_model.pth`：(訓練後生成) 儲存的模型權重檔案。
  * `training_loss_plot_*.png`：(訓練後生成) 訓練過程的視覺化圖表。
  * `Data/`：放置原始 HTML 或文字檔案的資料夾。
  * `2-cleaned_data.txt`：(生成) 資料清理後的輸出檔案。
  * `3-segmented_words.txt`：(生成) 分詞後的輸出檔案。
  * `4-word_to_id.txt`：(生成) 詞語對應 ID 的詞彙表。
  * `4-numerical_sequence.txt`：(生成) 訓練資料的數值化表示。

## 前置需求

在執行腳本前，請確保已安裝所有必要的函式庫：

```sh
pip install torch numpy matplotlib beautifulsoup4 pypdf requests jieba
```

## 使用方法

請按照以下步驟來執行整個專案流程：

### 1\. 資料準備

首先，你需要準備訓練資料。你可以選擇使用網頁、PDF 或多個 HTML 檔案。

**選項 A：下載單一網頁**
執行 `1-DLweb.py` 以下載 HTML 頁面，然後執行 `2-Data_Cleaning.py` 來提取文字。

**選項 B：處理多個 HTML 檔案**
將你的 HTML 檔案放入 `Data` 資料夾，然後執行 `2-pro-Data_Cleaning.py`。

**選項 C：從 PDF 提取文字**
將你的 PDF 檔案放入專案目錄中，並在 `2-PDF.py` 中更新 `input_pdf_file`、`start_page_num` 和 `end_page_num` 變數。接著執行該腳本。

### 2\. 中文分詞

選擇合適的腳本進行分詞。

若需要基礎分詞，執行 `3-Word_Segmentation.py`。若需要更強大、能正確處理特殊標記的分詞，請使用 `3-pro-Word_Segmentation.py`。

### 3\. 詞彙表與數值轉換

執行 `4-Vocabulary.py` 以建立詞彙表並將分詞後的文字轉換為數值序列。

### 4\. 模型訓練

在 `config.json` 中設定超參數，然後執行 `5-Model_Train.py` 進行模型訓練。訓練完成後，模型權重會儲存為 `simple_lstm_model.pth`，並會產生一個訓練曲線圖。

### 5\. 文字預測

模型訓練完成後，你可以使用 `6-Model_Predict.py` 來進行文字預測。在腳本中修改 `prompt` 變數為你想要預測的文字，然後執行腳本。
Configure your hyperparameters in `config.json` and then run `5-Model_Train.py` to train the model. The trained model weights will be saved as `simple_lstm_model.pth`, and a training plot will be generated.

### 5\. Text Prediction

Once the model is trained, you can use it to predict the next word. Update the `prompt` variable in `6-Model_Predict.py` with your desired text and run the script.
