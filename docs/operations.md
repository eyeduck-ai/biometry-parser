# 設定、模型與部署

## 安裝邊界

核心提供結構化PDF、影像讀取、資料契約、欄位解析、跨頁合併與API接頭。本機OCR runtime使用選用依賴 `[ocr]`；權重另行明確下載，不隨套件散布。原生PDF可在沒有OCR runtime及權重的環境使用。

```shell
python -m pip install .
python -m pip install ".[ocr]"
python -m biometry_parser models download ppocr-v6-small
```

本機路徑使用PaddleOCR的PP-OCRv6 small權重、RapidOCR及ONNX Runtime CPU；不需要PaddlePaddle、Torch或雲端帳號。模型授權以其來源為準。若選用依賴或權重缺失，影像辨識會回報可辨識的執行問題，不改用未授權的雲端路徑。

下載與解析使用相同的cache位置：

```shell
python -m biometry_parser models download ppocr-v6-small --cache-dir ./models
```

```python
from biometry_parser import BiometryParser
parser = BiometryParser(model_cache="./models", threads=2, timeout_seconds=600)
```

也可設定 `BIOMETRY_PARSER_MODEL_CACHE`。未設定時使用使用者的預設cache，不依賴本repo內的研究資料夾。模型會按manifest的SHA-256檢查，雜湊錯誤不會繼續推論。

## 高階介面

```python
from biometry_parser import BiometryParser

parser = BiometryParser(
    backend="local",
    timeout_seconds=600,
    threads=2,
)
result = parser.parse("report.pdf", page_strategy="biometry_first")
```

`backend` 為local、native、gemini或paddle_api。Gemini預設`gemini-3.5-flash-lite`，可設定`model="gemini-3.6-flash"`；Paddle API預設PP-OCRv6。指定API不代表文字層也要上傳：原生優先仍逐頁執行。

雲端必須明確設定 `allow_cloud=True`。即使所選文件最後全部能在本機讀取，選擇遠端workflow仍須先取得這個明確設定；套件不靠檔案內容決定你是否已授權傳輸。

```python
# 應用程式在建立parser前，已完成使用者授權及輸入資料處理。
parser = BiometryParser(backend="gemini", allow_cloud=True)
```

API key從`GEMINI_API_KEY`／`PADDLEOCR_ACCESS_TOKEN`或`api_key=`取得，只用於執行時。套件不自動讀.env、不把key存入config／JSON、不因doctor、匯入套件或建立設定而傳送報告。使用者仍應避免在自己的應用程式記錄完整設定、密鑰、病人原文或未遮罩圖片。

kit不會自動遮蔽個資。雲端輸入可能包含病人欄位、照片及生物特徵；應用端須自行完成適當的資料處理與傳輸授權。模型服務的額度、保留政策與價格由供應商決定，沒有把未知費用當作零的保證。

## TOML 與低階 API

`examples/config.toml` 提供完整、無密鑰的資料設定，可選workflow並建立其他專案的固定解析管線。

設定中的自訂模型、worker 與 manifest 相對路徑以呼叫程式的工作目錄為準；其他專案引用時可傳絕對路徑，或使用預設的每使用者模型快取。manifest 內的圖片相對路徑則以 manifest 所在目錄為準。

```python
from biometry_parser import extract, load_config

config = load_config("examples/config.toml")
result = extract("report.pdf", config, workflow="local")

# 選用遠端workflow仍須明確放行。
result = extract("prepared-scan.png", config, workflow="gemini", allow_cloud=True)
```

公開`extract()`也接受設定檔路徑或`ExtractionConfig`；可傳入`pages`、`page_strategy`及`progress`回呼。`env_overrides`可提供只傳至辨識子程序的執行時環境，避免改動主程序的全域環境；不要將其內容寫入log。

`Workflow.mode="single"` 保留單路候選；需要獨立來源檢查時，可用多個profile的crosscheck workflow。不同Paddle變體／API不是天然獨立來源。設定檔是資料，不會從輸入文件載入Python程式。

## 選頁

- `all`：預設全檔，適合完整擷取及跨頁差異檢查。
- `auto`：涵蓋原生摘要不同事實，保留IOL／未確定頁；影像只對完全相同像素去重。
- `biometry_first`：先挑原生雙眼資料完整的量測主頁，保留補頁、身分差異及IOL計算頁；重複的IOL頁量測摘要不再覆蓋主頁。
- `pages=[...]`：明確指定原始頁碼，從1起算；不會重新編號來源頁。

只需要量測且已有可排名的原生頁時，可先取得plan，再用其中量測頁呼叫解析：

```python
from biometry_parser import BiometryParser
from biometry_parser.page_selection import plan_pages

plan = plan_pages("report.pdf", strategy="biometry_first")
pages = plan["biometry_pages"] if plan["primary_biometry_page"] else plan["selected_pages"]
result = BiometryParser().parse("report.pdf", pages=pages)
```

這樣不要求其他頁的IOL表；若同時需要計算表，直接使用`page_strategy="biometry_first"`。掃描頁未有可用文字前，kit不會聲稱已知道哪頁資訊最多。原生選頁也不保證某OCR會讀齊；本版沒有自動重跑其他頁來補該模型的漏讀。

## 受檢核的替代圖片

進階profile可設定`options.input_manifest`，用應用端已審核的PNG替代原頁進行遠端辨識。manifest按原始文件SHA-256與原始頁碼綁定，並檢查檔案SHA、尺寸、單一PNG frame、已審核旗標與目錄範圍。這是完整性檢核，**不是自動去識別器或醫療隱私判定器**。

```toml
[profiles.gemini_lite.options]
input_manifest = "./prepared/manifest.json"
```

manifest概念範例如下；內容的SHA、尺寸、頁數和圖片必須由呼叫端產生並審核，不能直接使用此示意：

```json
{
  "kind": "reviewed_page_copies",
  "cases": [{
    "source_sha256": "<sha256-of-original-input>",
    "source_page_count": 1,
    "excluded_fields": ["patient_id", "gender", "measurement_date"],
    "pages": [{
      "page_number": 1,
      "image_path": "page-1.png",
      "image_sha256": "<sha256-of-reviewed-png>",
      "width": 2480,
      "height": 3508,
      "reviewed": true
    }]
  }]
}
```

圖片相對路徑必須位於manifest所在資料夾內，並符合實際辨識DPI下的原頁尺寸。只有確實從圖片移除的欄位才可列為excluded_fields；沒有清單不代表資料已匿名。

## 部署行為

OCR在隔離子程序內執行，以限制ML runtime對主程式記憶體與標準輸出的影響；API也採相同接頭契約。模型不會在套件import時載入。呼叫端應根據自己的併發量控制工作數；CPU threads是每個辨識器的設定，不是整個服務的全域並行上限。

回傳資料中的耗時、CPU／RSS與API用量只代表可測量範圍；遠端模型RAM不會以客戶端記憶體冒充。原始辨識文字與座標也可能包含敏感資訊，結果JSON應由應用端自行管理存取與保存。
