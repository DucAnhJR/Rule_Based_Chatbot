import pandas as pd
import json

# Phân tích dữ liệu training
print("=== PHÂN TÍCH DỮ LIỆU TRAINING ===")
df_data = pd.read_excel('data.xlsx')
print(f"Data shape: {df_data.shape}")
print(f"Columns: {df_data.columns.tolist()}")
print("\nFirst 10 rows:")
print(df_data.head(10))
print("\nSample answers:")
for i in range(min(5, len(df_data))):
    print(f"Q{i+1}: {df_data.iloc[i]['question']}")
    print(f"A{i+1}: {df_data.iloc[i]['answer']}")
    print("-" * 50)

# Phân tích dữ liệu test
print("\n=== PHÂN TÍCH DỮ LIỆU TEST ===")
df_test = pd.read_excel('test.xlsx')
print(f"Test shape: {df_test.shape}")
print(f"Columns: {df_test.columns.tolist()}")
print("\nFirst 10 rows:")
print(df_test.head(10))
print("\nSample test cases:")
for i in range(min(5, len(df_test))):
    print(f"Q{i+1}: {df_test.iloc[i]['question']}")
    print(f"A{i+1}: {df_test.iloc[i]['answer']}")
    if 'context' in df_test.columns:
        print(f"Context: {df_test.iloc[i]['context'][:100]}...")
    print("-" * 50)

# Phân tích độ dài câu hỏi và trả lời
print("\n=== PHÂN TÍCH ĐỘ DÀI ===")
print("Training data:")
print(f"Avg question length: {df_data['question'].str.len().mean():.2f}")
print(f"Avg answer length: {df_data['answer'].str.len().mean():.2f}")
print(f"Max question length: {df_data['question'].str.len().max()}")
print(f"Max answer length: {df_data['answer'].str.len().max()}")

print("\nTest data:")
print(f"Avg question length: {df_test['question'].str.len().mean():.2f}")
print(f"Avg answer length: {df_test['answer'].str.len().mean():.2f}")
print(f"Max question length: {df_test['question'].str.len().max()}")
print(f"Max answer length: {df_test['answer'].str.len().max()}")

# Phân tích từ khóa phổ biến
print("\n=== PHÂN TÍCH TỪ KHÓA ===")
import re
from collections import Counter

def get_keywords(text):
    # Tách từ và loại bỏ từ dừng đơn giản
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = {'và', 'của', 'có', 'là', 'trong', 'với', 'để', 'một', 'các', 'được', 'khi', 'về', 'như', 'hay', 'hoặc', 'từ', 'tại', 'theo', 'cho', 'trên', 'dưới', 'qua', 'ra', 'vào', 'mà', 'nào', 'nào', 'gì', 'sao', 'thế', 'thì', 'đã', 'sẽ', 'đang', 'phải', 'cần', 'nên', 'chỉ', 'cũng', 'đều', 'đã', 'và', 'hoặc', 'thì', 'nhưng', 'vì', 'do', 'bởi', 'vậy', 'nên', 'hay', 'mà', 'là', 'thành', 'làm', 'lên', 'xuống', 'bên', 'giữa', 'trong', 'ngoài', 'trước', 'sau', 'cuối', 'đầu', 'giữa', 'này', 'đó', 'kia', 'đây', 'đấy', 'ở', 'tới', 'đến', 'về', 'từ', 'sang', 'qua', 'lại', 'đi', 'tới', 'đến', 'về', 'cho', 'với', 'cùng', 'giống', 'khác', 'như', 'bằng', 'hơn', 'kém', 'nhất', 'cùng', 'cả', 'tất', 'mọi', 'nhiều', 'ít', 'đủ', 'thiếu', 'đầy', 'rỗng', 'tốt', 'xấu', 'đẹp', 'có', 'không', 'chưa', 'đã', 'sẽ', 'đang', 'từng', 'mới', 'cũ', 'trẻ', 'già', 'lớn', 'nhỏ', 'dài', 'ngắn', 'cao', 'thấp', 'rộng', 'hẹp', 'nhanh', 'chậm'}
    return [w for w in words if w not in stopwords and len(w) > 2]

# Từ khóa trong câu hỏi training
all_questions = ' '.join(df_data['question'].astype(str))
question_keywords = get_keywords(all_questions)
print("Top 20 từ khóa trong câu hỏi training:")
print(Counter(question_keywords).most_common(20))

# Từ khóa trong câu hỏi test
all_test_questions = ' '.join(df_test['question'].astype(str))
test_keywords = get_keywords(all_test_questions)
print("\nTop 20 từ khóa trong câu hỏi test:")
print(Counter(test_keywords).most_common(20))

print("\n=== PHÂN TÍCH SỰ TƯƠNG ĐỒNG ===")
training_keywords = set(question_keywords)
test_keywords_set = set(test_keywords)
common_keywords = training_keywords.intersection(test_keywords_set)
print(f"Số từ khóa chung: {len(common_keywords)}/{len(test_keywords_set)}")
print(f"Tỷ lệ từ khóa chung: {len(common_keywords)/len(test_keywords_set)*100:.2f}%")
