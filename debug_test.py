import pandas as pd
import requests
import json

# Test một số câu hỏi cụ thể để debug
test_questions = [
    "Những ngành học tại FPTU",
    "Sinh viên trễ bao nhiêu phút thì không được vào phòng thi?",
    "Đại học FPT có những ngành học nào?",
    "Các hành vi nào bị cấm trong phòng thi?",
    "Học phí của ngành công nghệ thông tin là bao nhiêu?"
]

API_URL = 'http://localhost:8000'

print("=== TEST DEBUG CÁC PHƯƠNG PHÁP MATCHING ===")

for i, question in enumerate(test_questions, 1):
    print(f"\n{i}. Câu hỏi: {question}")
    try:
        response = requests.post(API_URL, json={'chatInput': question}, timeout=30)
        if response.status_code == 200:
            result = response.json()
            print(f"   Trả lời: {result.get('output', 'Không có output')[:100]}...")
        else:
            print(f"   Lỗi HTTP: {response.status_code}")
    except Exception as e:
        print(f"   Lỗi: {e}")
    print("-" * 80)
