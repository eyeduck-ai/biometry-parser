# 輸出與資料契約

`BiometryParser.parse()` 回傳 `ExtractionResult`（Pydantic model），可使用 `model_dump(mode="json")` 或 `model_dump_json()`。輸出 schema 版本為1.2，可用 CLI 匯出 JSON Schema。

```shell
python -m biometry_parser schema -o extraction-result.schema.json
```

## 主要欄位

| 路徑 | 用途 |
|---|---|
| `source_sha256` | 原始輸入指紋 |
| `patient_id`、`gender`、`measurement_date` | 文件層級欄位 |
| `eyes.OD`、`eyes.OS` | 雙眼量測欄位 |
| `iol_calculations[]` | 各來源的 IOL 計算表 |
| `runs[]` | 各辨識路徑的觀察、IOL 表格、問題及資源記錄 |
| `issues[]` | 文件或欄位問題 |
| `metadata` | 選頁、分流與處理來源等記錄 |

每個一般量測 `FieldResult` 包含 `candidate_value`、`value`、`unit`、`data_status`、`verification_status` 及 `candidates[]`。Python 內以 Decimal 處理數字；JSON 保留可交換的數值表示。K值使用D，軸向degree，AL／ACD／LT／WTW／Pupil使用mm，CCT使用um。

```python
field = result.eyes["OS"].tk1.axis
if field.data_status.value == "not_measured":
    print("報告明確表示未量測")
elif field.candidate_value is not None:
    print("待檢核候選", field.candidate_value)
```

`value=None` 可能代表有候選但尚未通過交叉檢核，不是「原報告沒資料」。單一路徑，即使是原生PDF，也不自動成為臨床正式值。

## 七種資料狀態

| 狀態 | 意義 |
|---|---|
| `present` | 有可辨識數值或文字 |
| `not_measured` | 有明確未量測標記，例如整項`---` |
| `blank` | 有充分證據確認已標示的值區是空白 |
| `absent` | 有充分證據確認檢查範圍沒有此欄位 |
| `not_found` | 尚未找到答案，不證明欄位不存在 |
| `unreadable` | 找到欄位／值區，但無法可靠讀出 |
| `conflict` | 實際數值或明確未量測狀態互相矛盾 |

`blank`／`absent` 不能由模型回傳null或單純找不到標籤確認。處理失敗也不能算成欄位確定空白。整項TK未量測的配對軸向狀態，只在受限報告規則與完整標記證據成立時衍生，保留旗標及同一來源家族，沒有產生新的獨立模型票數。

資料狀態與 `verification_status` 分開；跨模型一致還必須符合獨立性、身分與路徑完整性條件。同一家族的Paddle本機變體與Paddle API不能僅因名稱不同就當作獨立證據。

## 來源與合併

`candidates[]` 內每個 observation 保存原始頁碼、眼別、原文、標籤、正規化bbox、模型、profile、family及flags。bbox採原頁左上角為原點，x/y介於0與1；無可靠位置時可為null。模型的bbox仍可能錯誤，保留座標不代表它已被人工確認。

跨頁保留有值的觀察，不讓另一頁漏讀或未列出覆蓋它；遇到真正數字差異或值／未量測矛盾時保持衝突。沒有任意平均或靜默挑選勝出模型。

## IOL 計算表

`iol_calculations[]` 及 `runs[].iol_calculations[]` 使用相同typed表格結構。每張表保持自己的鏡片、公式、常數、眼別與頁碼，不只按鏡片名稱跨頁覆蓋。

| 表格欄位 | 意義 |
|---|---|
| `table_id`、`page_number`、`bbox` | 表格識別與來源 |
| `eye`、`lens_model`、`lens_type` | 眼別與可確認的鏡片資訊 |
| `formula`、`constants`、`keratometry_source` | 公式、印出的常數、K／TK依據 |
| `target_refraction` | 明確印出的目標屈光度 |
| `rows[]` | 各候選IOL／預估屈光數據 |
| `verification_status` | 表格是否已檢核；預設unverified |

列欄位包含 `iol_power`、`predicted_refraction`、`toric_power`、`axis`、`row_kind`、`selection`、`highlight_status`、來源原文、bbox及flags。數值單位為D，axis為degree。

`target_refraction` 與每列的 `predicted_refraction` 是不同資料。明確的plano目標可正規化為0；只有Emmetropia字樣而未印數字時，不補造0。

`highlight_status` 描述灰底／粗體等版面提示，`selection` 則需要明確選定證據；灰底、中央列及接近plano均不等於已選擇植入。原生快速路徑不為讀灰底而額外渲染，影像獨有的提示可維持unknown。
