# pip install pypdf
import pypdf

def extract_text_from_pdf(input_pdf, output_txt, start_page, end_page):
    """
    從 PDF 的指定頁面範圍提取文字，並儲存到 .txt 檔案中。
    頁數從 0 開始計算。
    """
    try:
        with open(input_pdf, 'rb') as file:
            reader = pypdf.PdfReader(file)
            
            # 檢查頁面範圍是否有效
            if start_page < 0 or end_page >= len(reader.pages) or start_page > end_page:
                print("錯誤：您指定的頁面範圍無效。")
                print(f"此 PDF 檔案總共有 {len(reader.pages)} 頁（從 0 開始編號）。")
                return

            with open(output_txt, 'w', encoding='utf-8') as output_file:
                for page_num in range(start_page, end_page + 1):
                    page = reader.pages[page_num]
                    text = page.extract_text()
                    output_file.write(text)
                    print(f"已成功從第 {page_num + 1} 頁提取文字。")
        
        print(f"\n文字提取完成！內容已儲存到 '{output_txt}'。")
    
    except FileNotFoundError:
        print(f"錯誤：找不到檔案 '{input_pdf}'。請檢查檔案路徑。")
    except Exception as e:
        print(f"發生錯誤：{e}")

# 檔案名稱和頁面範圍設定
input_pdf_file = "90nchu0447003.pdf"
output_txt_file = "2-cleaned_data.txt"

# ❗ 重要：請修改這裡的起始頁和結束頁。
# 頁碼從 0 開始，例如，如果您想從第 1 頁到第 10 頁，請輸入 0 和 9。
start_page_num = 0  # 替換為您想開始的頁碼（-1 即可）。
end_page_num = 73  # 替換為您想結束的頁碼（-1 即可）。

# 6G的顯卡大約最大到73頁

# 執行函式
extract_text_from_pdf(input_pdf_file, output_txt_file, start_page_num, end_page_num)