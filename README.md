# Rule-Based Chatbot System - Hệ thống Chatbot Dựa trên Luật

## Tổng quan hệ thống

Đây là hệ thống chatbot rule-based tiếng Việt được phát triển để đạt accuracy cao (>60%) trong việc trả lời các câu hỏi về giáo dục. Hệ thống sử dụng nhiều kỹ thuật matching kết hợp để tăng khả năng nhận diện và trả lời chính xác.

## Cấu trúc Project

```
d:\Python\Rule_Based\
├── app.py                      # Main chatbot application (Advanced)
├── optimized_app.py           # Optimized version
├── optimized_chatbot.py       # Enhanced optimized version
├── test_chatbot.py            # Test suite chính
├── analyze_data.py            # Phân tích dữ liệu
├── progressive_test.py        # Test accuracy từng bước
├── debug_test.py              # Test debug nhanh
├── setup_elasticsearch.py     # Setup Elasticsearch
├── data.xlsx                  # Dữ liệu training
├── test.xlsx                  # Dữ liệu test
└── README.md                  # Tài liệu này
```

## Kiến trúc hệ thống

### 1. Core Components (app.py)

#### Class AdvancedChatbot
Chatbot chính với nhiều kỹ thuật matching:

**Khởi tạo:**
- Load dữ liệu từ `data.xlsx`
- Preprocessing text với nhiều bước chuẩn hóa
- Tạo multiple TF-IDF matrices cho các strategies khác nhau
- Xây dựng word frequency dictionary

**Preprocessing Pipeline:**
```python
def preprocess_text(self, text: str) -> str:
    """Enhanced preprocessing với better synonym handling"""
    # Chuẩn hóa Unicode
    # Xử lý từ viết tắt (sv → sinh viên, fptu → fpt, ojt → thực tập)
    # Synonym replacement
    # Loại bỏ punctuation và spaces dư thừa
```

**Normalize Question:**
```python
def normalize_question(self, text: str) -> str:
    """Enhanced normalize câu hỏi để tăng khả năng match"""
    # Chuẩn hóa question patterns
    # Xử lý từ hỏi (như thế nào, ở đâu, khi nào)
    # Content normalization
    # Loại bỏ redundant words
```

### 2. Matching Strategies

#### 2.1 Top-K Semantic Search
```python
def top_k_semantic_search(self, user_input: str, k: int = 8) -> List[Tuple[int, float]]:
    """Top-K semantic search với multiple strategies"""
```

**Hoạt động:**
- Sử dụng 3 TF-IDF matrices khác nhau:
  - Matrix 1: Processed questions + keywords (ngram 1-3)
  - Matrix 2: Normalized questions (ngram 1-2)
  - Matrix 3: Original questions (ngram 2-4, focus on phrases)
- Tính cosine similarity cho mỗi matrix
- Aggregate scores với weights và bonuses
- Trả về top-k candidates

#### 2.2 Exact Match
```python
def exact_match(self, user_input: str) -> Optional[str]:
    """Exact matching với nhiều variants"""
```

**Hoạt động:**
- Kiểm tra exact match với processed, normalized, và original text
- Ưu tiên cao nhất trong pipeline

#### 2.3 Advanced Keyword Match
```python
def advanced_keyword_match(self, user_input: str) -> Optional[str]:
    """Advanced keyword matching với context-aware scoring"""
```

**Hoạt động:**
- **Context-aware scoring:** Xác định context combinations (vd: "học" + "phí")
- **Important words:** Weighted scoring cho các từ quan trọng
- **Coverage scoring:** Tỷ lệ từ khóa được match
- **Keyword column matching:** Sử dụng cột keywords trong data
- **Partial matching:** Substring matching cho từ dài
- **Question type matching:** Nhận diện pattern câu hỏi
- **Structure similarity:** Độ tương đồng về cấu trúc

#### 2.4 Wildcard Match
```python
def wildcard_match(self, user_input: str) -> Optional[str]:
    """Enhanced wildcard matching cho flexible pattern recognition"""
```

**Hoạt động:**
- Sử dụng regex patterns cho các loại câu hỏi thường gặp
- 12 wildcard patterns với categories:
  - education (ngành học)
  - cost (học phí)
  - exam_location (phòng thi)
  - time (thời gian)
  - rules (quy định)
  - etc.
- Pattern diversity bonus
- Keyword overlap scoring

#### 2.5 Phrase Match
```python
def phrase_match(self, user_input: str) -> Optional[str]:
    """Enhanced phrase matching với context-aware scoring"""
```

**Hoạt động:**
- **Important phrase matching:** Weighted scoring cho cụm từ quan trọng
- **Multi-level N-gram matching:** N-gram từ 2-7 với rarity bonus
- **Substring matching:** Long và reverse substring matching
- **Pattern matching:** Question word patterns
- **Semantic phrase matching:** Key concept matching
- **Structure similarity:** Position-based matching

#### 2.6 Fuzzy Match
```python
def fuzzy_match(self, user_input: str) -> Optional[str]:
    """Enhanced fuzzy matching với multiple similarity metrics"""
```

**Hoạt động:**
- **Jaccard similarity:** Intersection/Union
- **Containment similarity:** Bidirectional containment
- **Normalized Jaccard:** Trên normalized text
- **Dice coefficient:** 2*intersection/(len1+len2)
- **Combined similarity:** Weighted combination
- **Important word bonus:** Bonus cho exact matches của important words

#### 2.7 Hybrid Top-K Search
```python
def hybrid_top_k_search(self, user_input: str) -> Optional[str]:
    """Hybrid Top-K search với fuzzy + keyword reranking"""
```

**Hoạt động:**
- Lấy top-k candidates từ semantic search
- **Fuzzy component:** Multiple similarity metrics
- **Keyword component:** Important + regular + keyword column + phrase matches
- **Context component:** Length similarity + question type
- **Final scoring:** Weighted combination với bonuses
- **Dynamic threshold:** Adaptive threshold dựa trên score distribution

### 3. Response Generation

#### Main Response Method
```python
def get_response(self, user_input: str) -> str:
    """Enhanced response với hybrid top-k approach"""
```

**Luồng hoạt động:**
1. **Primary:** Hybrid Top-K search
2. **Fallback methods:** Weighted voting system
   - exact_match (weight: 1.0)
   - wildcard_match (weight: 0.95)
   - advanced_keyword_match (weight: 0.9)
   - phrase_match (weight: 0.8)
   - fuzzy_match (weight: 0.7)
3. **Final fallback:** Chạy từng method với threshold thấp
4. **Default response:** Thông báo không tìm thấy

### 4. Flask API

#### Endpoints
- `POST /`: Main chat endpoint
  - Input: `{"chatInput": "câu hỏi"}`
  - Output: `{"output": "câu trả lời"}`

#### Usage
```bash
# Start API server
python app.py api

# Test request
curl -X POST http://localhost:8000 \
  -H "Content-Type: application/json" \
  -d '{"chatInput": "Những ngành học tại FPTU"}'
```

## Testing System

### 1. Test Scripts

#### test_chatbot.py
```python
# Test complete với concurrent requests
# Đọc test.xlsx, gửi requests đến API
# Tính accuracy, xuất kết quả ra Excel
```

#### progressive_test.py
```python
# Test nhanh trên sample nhỏ
# Kiểm tra accuracy từng bước cải tiến
```

#### debug_test.py
```python
# Test debug cho các câu hỏi cụ thể
# Kiểm tra response của từng method
```

### 2. Test Data

#### data.xlsx
- Dữ liệu training
- Columns: question, answer, keyword
- ~500+ câu hỏi về giáo dục

#### test.xlsx
- Dữ liệu test
- Columns: question, answer
- ~200+ câu hỏi test

### 3. Performance Metrics

**Current Performance:**
- Baseline accuracy: ~45%
- Target accuracy: 60%+
- Response time: <2s per question
- Concurrent handling: 3 workers

## Optimization Strategies

### 1. Multiple TF-IDF Matrices
- Khác nhau về ngram range, max_features
- Khác nhau về input data (processed, normalized, original)
- Ensemble voting để tăng accuracy

### 2. Weighted Scoring System
- Important words có weight cao
- Context-aware scoring
- Pattern-based scoring
- Coverage-based scoring

### 3. Fallback Mechanisms
- Multi-level fallback
- Weighted voting
- Dynamic thresholds
- Adaptive scoring

### 4. Text Preprocessing
- Unicode normalization
- Synonym replacement
- Abbreviation expansion
- Question pattern normalization

## Key Features

### 1. Synonym Handling
```python
# Xử lý từ viết tắt đặc biệt
r'\b(ojt|OJT)\b': 'thực tập'
r'\b(sv|sinh viên)\b': 'sinh viên'
r'\b(fptu?|đại học fpt)\b': 'fpt'
```

### 2. Context-Aware Matching
```python
# Context weights cho keyword combinations
context_weights = {
    'ngành': {'học': 1.5, 'fpt': 1.3},
    'học': {'phí': 1.5, 'ngành': 1.3},
    'phí': {'học': 1.5, 'bao': 1.4}
}
```

### 3. Dynamic Thresholds
```python
# Adaptive threshold dựa trên score distribution
if top_score > second_score * 1.3:
    threshold = 0.15
else:
    threshold = 0.25
```

### 4. Pattern Recognition
```python
# Wildcard patterns cho các loại câu hỏi
{
    'pattern': r'\b(học phí|chi phí|phí|tiền).*(bao nhiêu|mấy|giá)',
    'keywords': ['học', 'phí', 'bao', 'nhiêu'],
    'weight': 25,
    'category': 'cost'
}
```

## Usage Instructions

### 1. Setup Environment
```bash
pip install pandas scikit-learn flask numpy unicodedata
```

### 2. Run Chatbot
```bash
# Test mode
python app.py

# API mode
python app.py api
```

### 3. Run Tests
```bash
# Full test
python test_chatbot.py

# Quick test
python progressive_test.py

# Debug test
python debug_test.py
```

### 4. Analyze Data
```bash
python analyze_data.py
```

## Performance Optimization

### 1. Caching Strategy
- Cache TF-IDF matrices
- Cache preprocessing results
- Cache frequent queries

### 2. Concurrent Processing
- Multi-threading cho batch testing
- Async processing cho API

### 3. Memory Management
- Efficient matrix operations
- Sparse matrix usage
- Garbage collection

## Troubleshooting

### 1. Common Issues

**Low Accuracy:**
- Check preprocessing steps
- Verify synonym mappings
- Adjust thresholds
- Add more training data

**Slow Response:**
- Optimize TF-IDF parameters
- Reduce matrix dimensions
- Implement caching

**Memory Issues:**
- Reduce max_features
- Use sparse matrices
- Implement batch processing

### 2. Debug Tools

**Logging:**
```python
print(f"Semantic score: {semantic_score}")
print(f"Fuzzy score: {fuzzy_score}")
print(f"Final score: {final_score}")
```

**Method Testing:**
```python
# Test individual methods
result = chatbot.exact_match(question)
result = chatbot.advanced_keyword_match(question)
result = chatbot.wildcard_match(question)
```

## Future Enhancements

### 1. Machine Learning Integration
- Neural networks for better semantic understanding
- Word embeddings (Word2Vec, FastText)
- Transformer models (BERT, PhoBERT)

### 2. Advanced NLP
- Named Entity Recognition
- Dependency parsing
- Sentiment analysis

### 3. Performance Improvements
- GPU acceleration
- Distributed processing
- Real-time learning

### 4. Feature Additions
- Multi-turn conversations
- Context awareness
- User personalization

## Conclusion

Hệ thống chatbot rule-based này được thiết kế để đạt accuracy cao thông qua:

1. **Multiple matching strategies** với weighted ensemble
2. **Advanced preprocessing** với synonym và normalization
3. **Context-aware scoring** với important words và patterns
4. **Fallback mechanisms** với dynamic thresholds
5. **Comprehensive testing** với performance metrics

Hệ thống có thể dễ dàng mở rộng và tùy chỉnh cho các domain khác nhau bằng cách thay đổi:
- Training data
- Synonym mappings
- Important words
- Wildcard patterns
- Scoring weights

---

*Tác giả: GitHub Copilot*
*Ngày cập nhật: 2025-07-04*
