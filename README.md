# biometry-parser

供其他 Python 專案引用的眼科生物量測解析 kit。目前內建 **ZEISS IOLMaster 700** 報告解析器，重點是眼別、欄位語意、單位、來源證據及缺值狀態；辨識模型是可替換的底層接頭。

輸入支援具有文字層的 PDF、掃描／圖片型 PDF，以及 PNG、JPEG、TIFF、BMP、WebP 等圖檔。核心不包含網頁工作站、模型權重、研究用報告或 API key。

## 處理流程

1. 逐頁檢查 PDF 文字層與已支援的報告版面。
2. 可可靠解析的原生頁直接讀文字及座標，不先渲染大圖或載入 OCR。
3. 原生資料不足或輸入為影像時，交給選定的 OCR／VLM。
4. 以 IOLMaster 欄位與幾何規則解析 OD／OS、標準化單位並保留來源；再進行跨頁合併。
5. 回傳 typed Python 物件或 JSON，IOL 計算表與生物量測分開儲存。

預設使用本機 **PP-OCRv6 small** 的 PaddleOCR 模型權重，透過 RapidOCR／ONNX Runtime 在 CPU 執行。這條路徑不需安裝 PaddlePaddle。另提供 Gemini API 與 PaddleOCR 官方 API；API 接頭使用標準 HTTP，不要求安裝廠商 SDK。

## 安裝與引用

Python 3.11 以上。於此 repo 根目錄安裝：

```shell
python -m pip install .
```

需要本機 OCR 時，額外安裝 runtime 並明確下載模型：

```shell
python -m pip install ".[ocr]"
python -m biometry_parser models download ppocr-v6-small
```

其他專案可直接安裝此 checkout：

```shell
python -m pip install "/path/to/biometry-parser[ocr]"
```

套件名稱為 `biometry-parser`，Python import 名稱為 `biometry_parser`。匯入套件或建立設定不會自動下載權重、載入 OCR 模型或發出 API 請求。模型檔不包含在 wheel；下載會檢查鎖定的 SHA-256。

## 最小用法

```python
from pathlib import Path
from biometry_parser import BiometryParser

parser = BiometryParser()  # PDF 原生優先；影像才使用本機 Small
result = parser.parse("report.pdf")

od_al = result.eyes["OD"].al
print(od_al.data_status, od_al.candidate_value, od_al.unit)
print(od_al.verification_status)
Path("result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
```

`candidate_value` 是擷取候選；`value` 是通過 kit 檢核條件後的正式欄位值。單模型結果維持 `unverified`，因此不要用 `value is None` 判斷模型是否讀到數字，也不要把候選自動當成臨床最終答案。

只解析文字層、不允許 OCR／API fallback：

```python
result = BiometryParser(backend="native").parse("report.pdf")
```

也可用CLI輸出同一份JSON：

```shell
python -m biometry_parser extract report.pdf -o result.json
python -m biometry_parser extract report.pdf --backend native -o result.json
```

CLI退出碼0表示成功產生可用擷取結果，**不表示已臨床驗證**；2表示部分／失敗／不支援等需檢查的結果，1表示輸入或設定例外。請以JSON內的狀態及issues作為判斷依據。

## 選擇辨識接頭

| `backend` | 預設辨識器 | 額外需求 |
|---|---|---|
| `local` | 本機 PP-OCRv6 small | `[ocr]` runtime 與已下載權重 |
| `native` | PDF 原生文字／座標 | 核心依賴即可 |
| `gemini` | `gemini-3.5-flash-lite` | API key，明確開啟雲端 |
| `paddle_api` | 官方 API `PP-OCRv6` | AI Studio access token，明確開啟雲端 |

Gemini 也可明確設定 `model="gemini-3.6-flash"`。模型可用性、額度與費率由服務商及專案設定決定；kit 不會替你升級付費或把未知費用視為零。

```python
from biometry_parser import BiometryParser

# 從 GEMINI_API_KEY 環境變數取得 key；只在影像 fallback 時呼叫 API。
gemini = BiometryParser(backend="gemini", allow_cloud=True)
result = gemini.parse("scan.png")

flash = BiometryParser(
    backend="gemini", model="gemini-3.6-flash", allow_cloud=True,
)

# 從 PADDLEOCR_ACCESS_TOKEN 環境變數取得 token。
paddle = BiometryParser(backend="paddle_api", allow_cloud=True)
```

也可透過 `api_key=` 傳入執行時密鑰。高階介面的雲端呼叫必須明確指定 `allow_cloud=True`；密鑰不寫入設定或結果。即使選擇雲端接頭，可靠的原生 PDF 頁仍先在本機解析。

此 repo 不自動載入 `.env`。`.env.example` 僅列環境變數名稱；由呼叫端的秘密管理方式提供密鑰。雲端輸入會離開本機，kit **不會自動去識別**，請在呼叫前處理。需要受檢核的逐頁替代圖片時，見[執行設定](docs/operations.md)。

## 擷取欄位

OD、OS 各自保存：

- AL、ACD、LT、CCT、WTW、Pupil／P。
- Mean K（SE／Avg. K）、角膜散光（ΔK／Ast. K）及可辨識的散光軸向。
- K1、K2、TK1、TK2 的屈光度與軸向。

另保存 Patient ID、Gender、Date of measurement。IOL 計算資料在 `iol_calculations[]`，包含鏡片名稱／種類、公式、常數、K／TK 來源、目標屈光度，以及各候選 IOL 度數和預估術後屈光度。每筆保留眼別、頁碼、位置、原文與來源模型。

灰底或粗體列與「明確選定」分開記錄；不以最接近零度的列推定植入選擇，也不從 Emmetropia 字樣補造未印出的數字。

## 選頁與跨頁合併

```python
# 原始頁碼，一律從 1 起算。
result = parser.parse("report.pdf", pages=[2, 4])

# 原生頁優先挑出完整量測摘要，IOL 計算頁另保留。
result = parser.parse("report.pdf", page_strategy="biometry_first")
```

預設 `all` 處理全檔。`auto` 保留原生資料的不同數值／狀態與 IOL 計算頁；影像只有完全相同像素才去重。`biometry_first` 以雙眼量測覆蓋率排序，CCT 作為條件之一；主頁缺資料時保留補頁，並另保留計算表。掃描檔在未辨識前無法可靠排名，本版保守保留影像頁，不宣稱已實作低解析度前篩。

某頁有值、其他頁未列出或漏讀時保留有值結果。不同實際數字，或實際數字與明確未量測互相矛盾，才建立衝突；不平均、不以多數票挑選。不同病人、日期與不確定眼別會保留問題供呼叫端處理。

## 資料狀態與擴充

資料層保存 `present`、`not_measured`、`blank`、`absent`、`not_found`、`unreadable`、`conflict` 七種狀態。資料狀態與執行是否成功、是否交叉檢核為不同概念。缺值不等於不存在，OCR 信心值也不等於欄位準確率。

- [輸出與資料契約](docs/data-contract.md)
- [設定、模型與部署](docs/operations.md)
- [增加辨識器或報告品牌](docs/extending.md)
- [獨立安裝與功能驗證](docs/validation.md)
- [最小本機範例](examples/local.py)／[雲端範例](examples/cloud.py)

目前只有 IOLMaster 700 的報告解析規則。註冊新模型不代表支援新品牌；新品牌必須提供自己的版面辨識及欄位解析器。
