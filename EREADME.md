<h1 align="center">CLanguage_Mode_for_2</h1>
<p align="center">CLRE-20</p>
<p align="center"><a href="/README.md">Chinese version</a> | <a href="/EREADME.md">English version</a></p>
---

# LSTM-Based Chinese Text Generation Model

This repository demonstrates a **Chinese text generation model** implemented with **PyTorch** using an **LSTM (Long Short-Term Memory)** network.
The pipeline covers **Data Collection → Cleaning → Segmentation → Vocabulary → Training → Text Generation**, making it a fundamental reference for building Chinese language models.

## Features

* **Data Cleaning:** Extracts text from HTML `<p>` tags or specified PDF page ranges.
* **Chinese Segmentation:** Uses `jieba` for tokenization with support for custom tokens (e.g., `[END_OF_NOVEL]`).
* **Vocabulary Creation:** Converts words into numerical IDs with `<unk>` and `<pad>` tokens.
* **Model Training:** `SimpleLSTM` trained with cross-entropy loss and Adam optimizer, including visualization of loss/accuracy curves.
* **Text Prediction:** Loads the trained model to generate text based on prompts.
* **Configuration:** All hyperparameters are centralized in `config.json`.

## Project Structure

| File / Folder                | Description                                 |
| ---------------------------- | ------------------------------------------- |
| `1-DLweb.py`                 | Download a single web page                  |
| `2-Data_Cleaning.py`         | Extract `<p>` text from a single HTML file  |
| `2-pro-Data_Cleaning.py`     | Process multiple HTML files in a folder     |
| `2-PDF.py`                   | Extract text from a PDF within a page range |
| `3-Word_Segmentation.py`     | Basic Chinese word segmentation             |
| `3-pro-Word_Segmentation.py` | Enhanced version with custom dictionary     |
| `4-Vocabulary.py`            | Build vocabulary and numerical sequences    |
| `5-Model_Train.py`           | Train the LSTM and plot training curves     |
| `6-Model_Predict.py`         | Predict text using the trained model        |
| `config.json`                | Hyperparameter configuration                |
| `Data/`                      | Raw HTML, PDF, or text data                 |
| `*.txt / *.png / *.pth`      | Generated outputs                           |

## Requirements

Install dependencies:

```bash
pip install torch numpy matplotlib beautifulsoup4 pypdf requests jieba
```

## Usage

1. **Data Preparation**

   * Single HTML → `1-DLweb.py` → `2-Data_Cleaning.py`
   * Multiple HTML files → `2-pro-Data_Cleaning.py`
   * PDF → edit parameters in `2-PDF.py` and run

2. **Word Segmentation**

   * Basic: `3-Word_Segmentation.py`
   * Advanced: `3-pro-Word_Segmentation.py`

3. **Vocabulary & Conversion**

   ```bash
   python 4-Vocabulary.py
   ```

4. **Model Training**

   ```bash
   python 5-Model_Train.py
   ```

   Outputs:

   * `simple_lstm_model.pth` → model weights
   * `training_loss_plot_*.png` → training visualization

5. **Text Prediction**
   Edit `prompt` in `6-Model_Predict.py` and run:

   ```bash
   python 6-Model_Predict.py
   ```
