# 套件驗證

2026-11-17，Windows／Python 3.11 的本機驗證：

- 建立乾淨虛擬環境，只安裝 wheel 與 test extra，在 checkout 外執行 **345 項測試，全部通過**。
- 該環境沒有舊專案套件，也沒有 RapidOCR／ONNX Runtime；原生 PDF 仍能取得 24 張、144 列 IOL 表格，摘要優先模式的眼科量測來源為第 6 頁，沒有渲染 PNG 或啟動 OCR。
- 另行安裝 `[ocr]`，使用鎖定的 PP-OCRv6 small 權重，實測掃描 PDF 與 PNG 圖片皆經本機 OCR 取得雙眼核心欄位及明確的 TK 未量測狀態。
- API 接頭、密鑰隔離、雲端 opt-in、回應驗證以 mock／合成資料測試；本次沒有發出雲端辨識請求。
- 安裝檔包含原始碼、模型 manifest 與資料 schema；不含工作站、權重、使用者報告或 API key。

實測所用私人報告沒有放進 repo、測試或安裝檔。這些檢查證明套件拆分與指定路徑可運作，不代表所有 IOLMaster 700 版本都已取得臨床驗證。

GitHub Actions 已設定 Windows／Ubuntu、Python 3.11／3.12，會在建置並安裝 wheel 後，將測試搬到 checkout 外執行。本機驗證不等於這些遠端 CI 工作已執行。
