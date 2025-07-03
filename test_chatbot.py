import concurrent.futures
import time
import pandas as pd
import requests
import os

# Đường dẫn file test và file kết quả
input_file = 'test.xlsx'  # ĐÂY PHẢI LÀ FILE TEST
output_file = os.path.join(os.path.dirname(input_file), 'test_result.xlsx')

# Đọc file test
test_df = pd.read_excel(input_file)

API_URL = 'http://localhost:8000'

results = []

def send_request(row):
    question = str(row['question']).strip()
    expected = str(row['answer']).strip()   # Đáp án đúng để kiểm tra
    try:
        response = requests.post(API_URL, json={'chatInput': question}, timeout=90)  # Tăng timeout lên 90s
        response.raise_for_status()
        chatbot_ans = response.json().get('output', '').strip()
    except requests.exceptions.Timeout:
        chatbot_ans = "Lỗi: Timeout - Server phản hồi chậm"
    except requests.exceptions.ConnectionError:
        chatbot_ans = "Lỗi: Không thể kết nối đến server"
    except Exception as e:
        chatbot_ans = f"Lỗi: {e}"
    is_correct = chatbot_ans.lower() == expected.lower()
    return {
        'Câu hỏi': question,
        'Đáp án đúng': expected,
        'Chatbot trả lời': chatbot_ans,
        'Đúng/Sai': 'Đúng' if is_correct else 'Sai',
        'is_correct': is_correct
    }

start_time = time.time()

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:  # Giảm workers để giảm tải
    futures = [executor.submit(send_request, row) for idx, row in test_df.iterrows()]
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        results.append(result)

correct = sum(r['is_correct'] for r in results)
total = len(test_df)
accuracy = (correct / total * 100) if total > 0 else 0

results.append({
    'Câu hỏi': 'Tổng kết',
    'Đáp án đúng': '',
    'Chatbot trả lời': f'{correct}/{total}',
    'Đúng/Sai': f'Tỉ lệ đúng: {accuracy:.2f}%'
})

result_df = pd.DataFrame(results)
result_df.to_excel(output_file, index=False)

end_time = time.time()

print(f"Số câu đúng: {correct}/{total}")
print(f"Tỉ lệ đúng: {accuracy:.2f}%")
print(f"Đã xuất kết quả ra file {output_file}")
print(f"Tổng thời gian chạy: {end_time - start_time:.2f} giây")