# 增加辨識器或報告品牌

辨識模型與報告解析器是兩層介面。新OCR可以沿用IOLMaster欄位規則；新品牌則需自己的身分／版面辨識及解析器。註冊名稱不會自動讓新儀器可用。

## 辨識器

接頭提供`describe()`及`recognize(PageInput) -> ProviderOutput`。OCR回傳spans（原文、正規化bbox及可選原生confidence）；VLM可回傳claims與IOLTableClaim，由共用解析層正規化。

```python
from biometry_parser import register_provider
from my_app.adapters import create_provider

register_provider("my_ocr", create_provider)
```

factory必須是可匯入模組中的頂層函式，因為辨識在隔離子程序執行；不要使用lambda、區域函式或`__main__` factory。`examples/custom_provider.py`只展示契約，不代表新增辨識能力。

也可從你自己的套件提供entry point：

```toml
[project.entry-points."biometry_parser.providers"]
my_ocr = "my_app.adapters:create_provider"
```

接頭和plugin是應用端明確安裝的可信任程式，不能從報告文字、JSON或其他輸入內容載入。未知接頭的傳輸行為無法由kit推斷；公開API可能要求明確放行，應用端須審核自訂程式。

`describe()`至少應明確描述id、family、kind及remote。family應代表真實模型來源，不能為同一權重任意換名字，藉此產生虛假的獨立驗證。原生confidence的尺度也應保留，不能當作欄位正確率。

接頭只複製證據，不要求模型自行裁決七種最終狀態。明確`---`可回報not_measured；找不到是not_found，有標籤但讀不出是unreadable。空白／缺欄需要更多證據，跨頁conflict與verification由kit判斷。IOL灰底不等於選定。

## 報告解析器

```python
from biometry_parser import register_report_template
from biometry_parser.report_templates import ReportTemplate
from my_app.report_format import matches_report, parse_report_page

register_report_template(ReportTemplate(
    id="my_biometer",
    manufacturer="Example",
    device="Biometer X",
    matcher=matches_report,
    parser=parse_report_page,
    version="1",
))
```

matcher接受可辨識文字，應明確辨別儀器而非僅包含一個常見欄位。parser接收page、ProviderOutput，以及profile_id／model_id／family／date_order等參數，回傳ParsedPage。它必須保留真正的眼別、頁碼、原文、bbox與問題，無法確認時回傳不支援或待檢核資料。

僅有matcher、沒有parser不會標成可用；多個模板同時符合也不能任意選一個。未知品牌不能套用「左半OD、右半OS」捷徑：單眼頁、旋轉、不同表格順序與標題位置均需測試。

新品牌至少應建立不含個資的合成測試，涵蓋雙眼／單眼、缺值、`---`、單位、不同日期／病人、跨頁差異、公式／鏡片表及錯位文字。原生PDF、OCR與VLM輸入都需依相同欄位契約驗證；不要用模型自己的輸出建立該模型的正確答案。
