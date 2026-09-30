# VGEC zero-shot Azure — type_id + fixed_sentence

Bản chia sẻ tối giản để chạy lại pipeline zero-shot đã dùng cho `azure-4.1` và `azure-4o`. Model nhận **chỉ prompt và câu input**; ID, gold label và reference không được gửi tới API. Output bắt buộc là JSON có đúng hai trường:

```json
{"type_id": 1, "fixed_sentence": "Câu sau khi sửa."}
```

Đây là scorer nội bộ phục vụ tái lập thí nghiệm, không phải scorer chính thức của VLSP.

## 1. Cài đặt

Yêu cầu Python 3.10 trở lên.

```powershell
git clone <URL_REPOSITORY>
cd VGEC-share
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Mở `.env` và điền endpoint OpenAI-compatible cùng API key của bạn:

```dotenv
API_BASE_URL=https://your-openai-compatible-proxy.example.com
API_KEY=replace_with_your_api_key
```

Code tự nối `/v1/chat/completions` nếu endpoint chưa chứa đường dẫn này và gửi key bằng header `x-litellm-api-key`. Không commit `.env`.

## 2. Chạy ngay một câu tự tạo, không cần dataset

Dry-run hoàn toàn local, không gọi API:

```powershell
python -m vgec.predict --model azure-4.1 --input-file examples/example_sentence.txt --dry-run
```

Gọi thật sau khi đã cấu hình `.env`:

```powershell
python -m vgec.predict --model azure-4.1 --input-file examples/example_sentence.txt
python -m vgec.predict --model azure-4o --sentence "Qua tác phẩm cho thấy tình yêu quê hương của tác giả."
```

CLI in trực tiếp `type_id` và `fixed_sentence`. Ví dụ trong thư mục `examples/` là câu tự tạo, không lấy từ dữ liệu VGEC.

## 3. Chuẩn bị dữ liệu VGEC

Không có dữ liệu cuộc thi trong repository này. Khi đã được cấp dữ liệu, đặt file nguồn tại thư mục gốc, ví dụ `VGEC 2026_v2.json`, rồi chạy:

```powershell
python -m vgec.preprocess --input "VGEC 2026_v2.json"
```

Các mẫu có reference bất thường được đưa vào `data/reports/review_queue.csv` và tạm loại khỏi evaluation. Nếu cần duyệt, điền `data/reports/review_decisions.csv` theo bốn cột:

```text
id,action,references,reason
```

Trong đó `action` là `replace`, `exclude` hoặc `keep`; `references` là chuỗi/list JSON. Sau đó chạy lại preprocess. Không sửa trực tiếp `data/processed/eval_clean.jsonl`.

Runner khóa SHA-256 và thứ tự ID theo `data/reports/manifest.json`, nên không thể vô tình resume trên dataset khác.

## 4. Kiểm tra 5 mẫu

Dry-run local chọn một mẫu thuộc mỗi Type 0–4:

```powershell
python -m vgec.run --model azure-4.1 --limit 5 --sample-by-type --dry-run --run-id dryrun-azure41
python -m vgec.score --run-id dryrun-azure41 --allow-partial
```

Chạy thật 5 mẫu cho từng model:

```powershell
python -m vgec.run --model azure-4.1 --limit 5 --sample-by-type --run-id azure41-v2
python -m vgec.run --model azure-4o --limit 5 --sample-by-type --run-id azure4o-v2
```

Kiểm tra `runs/<run-id>/predictions.jsonl` trước khi mở rộng. Mỗi dòng thành công phải có `predicted_type_id` từ 0 đến 4, `prediction`, `raw_prediction` và `status=ok`.

## 5. Chạy full và resume

Giữ nguyên run ID của smoke test. Runner tự bỏ qua ID đã thành công:

```powershell
python -m vgec.run --model azure-4.1 --all --confirm-full-run --run-id azure41-v2
python -m vgec.run --model azure-4o --all --confirm-full-run --run-id azure4o-v2
```

Auth/quota làm run dừng ngay. HTTP 429 rate-limit, lỗi mạng và lỗi server tạm thời được retry exponential backoff. Chạy lại đúng lệnh để resume các ID chưa thành công.

## 6. Chấm điểm và report

```powershell
python -m vgec.score --run-id azure41-v2
python -m vgec.score --run-id azure4o-v2

python -m vgec.report --run-id azure41-v2 --run-id azure4o-v2 --results-dir results/azure-v2
```

Mỗi run sinh `metrics.json` và `errors.csv`. Report chỉ gộp các run có cùng dataset hash, scorer version, tập ID được chấm và prompt hash.

## 7. Prompt và protocol đã khóa

- Prompt: `prompts/v2_type_and_correction.json`
- Prompt version: `v2_type_and_correction`
- SHA-256 của file prompt đã dùng trong hai baseline: `d87dc880b72693761ea5ebd14880254a9003faa81382c6eb3e5faed055b304c6`
- `temperature=0`, `max_tokens=256`, JSON mode.
- Parser từ chối JSON có field thừa, `type_id` ngoài 0–4 hoặc `fixed_sentence` rỗng.
- Prediction, config và error log không lưu API key.

## 8. Test

```powershell
python -m unittest discover -s tests -v
python -m vgec.predict --model azure-4.1 --input-file examples/example_sentence.txt --dry-run
```

Các lệnh test và dry-run không gọi API, không tải model và không cần dữ liệu VGEC.
