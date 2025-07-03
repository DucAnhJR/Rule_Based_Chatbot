"""
Rule-Based Chatbot với Elasticsearch

Để sử dụng Elasticsearch:
1. Tải và cài đặt Elasticsearch từ: https://www.elastic.co/downloads/elasticsearch
2. Chạy Elasticsearch trên localhost:9200
3. Chạy script này sẽ tự động tải dữ liệu vào Elasticsearch

Nếu không có Elasticsearch, chatbot sẽ tự động chuyển sang sử dụng DataFrame
"""

import pandas as pd
import re
from typing import List, Dict, Optional
from elasticsearch import Elasticsearch
import json
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pyvi import ViTokenizer
import unicodedata

class RuleBasedChatbot:
    def __init__(self, data_file: str = 'data.xlsx', use_elasticsearch: bool = True):
        """Khởi tạo chatbot với dữ liệu từ file Excel và Elasticsearch"""
        self.use_elasticsearch = use_elasticsearch
        self.index_name = "chatbot_data"
        
        if self.use_elasticsearch:
            try:
                # Kết nối đến Elasticsearch (local instance)
                self.es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
                
                # Kiểm tra kết nối
                if not self.es.ping():
                    print("Cảnh báo: Không thể kết nối đến Elasticsearch. Sử dụng DataFrame thay thế.")
                    self.use_elasticsearch = False
                    self._load_data_to_dataframe(data_file)
                else:
                    print("Kết nối Elasticsearch thành công!")
                    self._setup_elasticsearch_index()
                    self._load_data_to_elasticsearch(data_file)
                    
            except Exception as e:
                print(f"Lỗi khi kết nối Elasticsearch: {e}")
                print("Sử dụng DataFrame thay thế.")
                self.use_elasticsearch = False
                self._load_data_to_dataframe(data_file)
        else:
            self._load_data_to_dataframe(data_file)
    
    def _load_data_to_dataframe(self, data_file: str):
        """Tải dữ liệu vào DataFrame"""
        self.data = pd.read_excel(data_file)
        self.data = self.data.dropna(subset=['question', 'answer'])
        self.data['question'] = self.data['question'].astype(str).str.lower()
        self.data['answer'] = self.data['answer'].astype(str)
        
        # Preprocessing cho semantic search với nhiều variations
        self.data['processed_question'] = self.data['question'].apply(self.full_preprocess)
        self.data['simple_processed'] = self.data['question'].apply(self.preprocess_text)
        self.data['important_words'] = self.data['question'].apply(lambda x: ' '.join(self.extract_important_words(x)))
        
        # Kết hợp tất cả variations để tạo corpus phong phú hơn
        all_questions = []
        for idx, row in self.data.iterrows():
            combined = f"{row['processed_question']} {row['simple_processed']} {row['important_words']}"
            all_questions.append(combined)
        
        # Tạo TF-IDF vectorizer với tham số tối ưu hơn
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,  # Tăng số features
            ngram_range=(1, 3),  # Bao gồm 1-gram, 2-gram, 3-gram
            min_df=1,           # Tần số tối thiểu
            max_df=0.95,        # Tần số tối đa
            lowercase=True,
            stop_words=None     # Không dùng stop_words built-in
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_questions)
        
        # Chỉ in thông tin khi không chạy API
        import sys
        if not (len(sys.argv) > 1 and sys.argv[1] == 'api'):
            print(f"Đã tải {len(self.data)} câu hỏi từ {data_file}")
            print(f"Đã tạo TF-IDF matrix với {self.tfidf_matrix.shape[1]} features")
    
    def _setup_elasticsearch_index(self):
        """Thiết lập index cho Elasticsearch"""
        index_mapping = {
            "mappings": {
                "properties": {
                    "question": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "answer": {
                        "type": "text"
                    },
                    "nguon": {
                        "type": "text"
                    }
                }
            }
        }
        
        # Xóa index cũ nếu tồn tại
        if self.es.indices.exists(index=self.index_name):
            self.es.indices.delete(index=self.index_name)
        
        # Tạo index mới
        self.es.indices.create(index=self.index_name, body=index_mapping)
    
    def _load_data_to_elasticsearch(self, data_file: str):
        """Tải dữ liệu từ Excel vào Elasticsearch"""
        df = pd.read_excel(data_file)
        df = df.dropna(subset=['question', 'answer'])
        
        # Đưa dữ liệu vào Elasticsearch
        for idx, row in df.iterrows():
            doc = {
                "question": str(row['question']).lower(),
                "answer": str(row['answer']),
                "nguon": str(row['nguồn']) if pd.notna(row['nguồn']) else ""
            }
            
            self.es.index(index=self.index_name, id=idx, body=doc)
        
        # Refresh index để đảm bảo dữ liệu có sẵn
        self.es.indices.refresh(index=self.index_name)
        print(f"Đã tải {len(df)} bản ghi vào Elasticsearch!")
        
    def preprocess_text(self, text: str) -> str:
        """Xử lý văn bản đầu vào với Text Normalization và Noise Removal"""
        if not text:
            return ""
        
        # Text Normalization
        text = str(text).lower().strip()
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Noise Removal - giữ lại nhiều ký tự hơn
        text = re.sub(r'[^\w\s]', ' ', text)  # Loại bỏ ký tự đặc biệt
        text = re.sub(r'\s+', ' ', text)      # Loại bỏ khoảng trắng thừa
        
        return text.strip()
    
    def tokenize_vietnamese(self, text: str) -> List[str]:
        """Tokenization cho tiếng Việt"""
        if not text:
            return []
        
        # Sử dụng pyvi để tách từ tiếng Việt
        try:
            tokenized = ViTokenizer.tokenize(text)
            tokens = tokenized.split()
            return [token for token in tokens if len(token) > 0]  # Giảm từ 1 xuống 0
        except:
            # Fallback nếu pyvi không hoạt động
            return text.split()
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Loại bỏ stopwords tiếng Việt - ít hơn để giữ nhiều từ quan trọng"""
        stop_words = {
            'là', 'của', 'và', 'có', 'được', 'này', 'đó', 'cho', 'với', 'về',
            'thì', 'mà', 'rồi', 'đang', 'vẫn', 'đều', 'cũng'
        }
        
        return [token for token in tokens if token not in stop_words and len(token) > 0]
    
    def extract_important_words(self, text: str) -> List[str]:
        """Trích xuất từ quan trọng không loại bỏ stopwords"""
        normalized = self.preprocess_text(text)
        tokens = self.tokenize_vietnamese(normalized)
        return [token for token in tokens if len(token) > 0]
    
    def full_preprocess(self, text: str) -> str:
        """Preprocessing đầy đủ: Normalization + Tokenization + Stopword removal"""
        # Text Normalization và Noise Removal
        normalized = self.preprocess_text(text)
        
        # Tokenization
        tokens = self.tokenize_vietnamese(normalized)
        
        # Remove stopwords
        clean_tokens = self.remove_stopwords(tokens)
        
        return ' '.join(clean_tokens)
    
    def wildcard_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm wildcard pattern - linh hoạt hơn"""
        input_words = self.extract_important_words(user_input)
        
        best_match = None
        max_matches = 0
        
        for idx, row in self.data.iterrows():
            question_words = self.extract_important_words(row['question'])
            
            # Đếm số từ khớp
            matches = 0
            for input_word in input_words:
                for question_word in question_words:
                    if len(input_word) > 1 and (input_word in question_word or question_word in input_word):
                        matches += 1
                        break
            
            # Cập nhật best match
            if matches > max_matches and matches > 0:
                max_matches = matches
                best_match = row['answer']
        
        # Trả về nếu có ít nhất 1 từ khớp
        return best_match if max_matches > 0 else None
    
    def fuzzy_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm fuzzy matching - ngưỡng thấp hơn"""
        input_words = set(self.extract_important_words(user_input))
        
        best_match = None
        best_similarity = 0
        
        for idx, row in self.data.iterrows():
            question_words = set(self.extract_important_words(row['question']))
            
            if input_words and question_words:
                # Tính similarity với nhiều cách khác nhau
                intersection = len(input_words & question_words)
                union = len(input_words | question_words)
                
                # Jaccard similarity
                jaccard = intersection / union if union > 0 else 0
                
                # Containment similarity
                containment = intersection / len(input_words) if len(input_words) > 0 else 0
                
                # Lấy max của 2 similarity
                similarity = max(jaccard, containment)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = row['answer']
        
        # Ngưỡng thấp hơn để dễ match hơn
        return best_match if best_similarity > 0.1 else None
    
    def match_phrase(self, user_input: str) -> Optional[str]:
        """Tìm kiếm match phrase - linh hoạt hơn"""
        processed_input = self.preprocess_text(user_input)
        input_words = processed_input.split()
        
        best_match = None
        max_score = 0
        
        for idx, row in self.data.iterrows():
            question = self.preprocess_text(row['question'])
            question_words = question.split()
            
            score = 0
            
            # Kiểm tra subsequence matching
            for i in range(len(input_words)):
                for j in range(i + 1, len(input_words) + 1):
                    phrase = ' '.join(input_words[i:j])
                    if len(phrase) > 2 and phrase in question:
                        score += len(phrase)
            
            # Kiểm tra reverse matching
            for i in range(len(question_words)):
                for j in range(i + 1, len(question_words) + 1):
                    phrase = ' '.join(question_words[i:j])
                    if len(phrase) > 2 and phrase in processed_input:
                        score += len(phrase)
            
            if score > max_score:
                max_score = score
                best_match = row['answer']
        
        return best_match if max_score > 0 else None
    
    def semantic_search(self, user_input: str) -> Optional[str]:
        """Tìm kiếm semantic - ngưỡng thấp hơn"""
        if not hasattr(self, 'tfidf_vectorizer') or not hasattr(self, 'tfidf_matrix'):
            return None
        
        # Preprocessing input với nhiều cách khác nhau
        processed_options = [
            self.full_preprocess(user_input),
            self.preprocess_text(user_input),
            ' '.join(self.extract_important_words(user_input))
        ]
        
        best_match = None
        best_similarity = 0
        
        for processed_input in processed_options:
            if not processed_input:
                continue
            
            try:
                # Tạo vector cho input
                input_vector = self.tfidf_vectorizer.transform([processed_input])
                
                # Tính cosine similarity
                similarities = cosine_similarity(input_vector, self.tfidf_matrix)
                
                # Lấy similarity cao nhất
                max_sim = np.max(similarities[0])
                
                if max_sim > best_similarity:
                    best_similarity = max_sim
                    best_match_idx = np.argmax(similarities[0])
                    best_match = self.data.iloc[best_match_idx]['answer']
            except:
                continue
        
        # Ngưỡng rất thấp để dễ match
        return best_match if best_similarity > 0.01 else None
    
    def keyword_overlap_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm dựa trên số lượng từ khóa chung"""
        input_words = set(self.extract_important_words(user_input))
        
        best_match = None
        max_overlap = 0
        
        for idx, row in self.data.iterrows():
            question_words = set(self.extract_important_words(row['question']))
            
            # Đếm số từ chung
            overlap = len(input_words & question_words)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = row['answer']
        
        # Trả về nếu có ít nhất 1 từ chung
        return best_match if max_overlap > 0 else None
    
    def partial_string_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm chuỗi con"""
        processed_input = self.preprocess_text(user_input)
        
        # Tìm câu hỏi chứa input hoặc ngược lại
        for idx, row in self.data.iterrows():
            question = self.preprocess_text(row['question'])
            
            # Kiểm tra chuỗi con với độ dài tối thiểu
            if len(processed_input) > 2 and processed_input in question:
                return row['answer']
            
            if len(question) > 2 and question in processed_input:
                return row['answer']
        
        return None
    
    def find_exact_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm khớp chính xác"""
        if self.use_elasticsearch:
            return self._find_exact_match_es(user_input)
        else:
            return self._find_exact_match_df(user_input)
    
    def _find_exact_match_df(self, user_input: str) -> Optional[str]:
        """Tìm kiếm khớp chính xác bằng DataFrame"""
        processed_input = self.preprocess_text(user_input)
        
        for idx, row in self.data.iterrows():
            if processed_input == self.preprocess_text(row['question']):
                return row['answer']
        return None
    
    def _find_exact_match_es(self, user_input: str) -> Optional[str]:
        """Tìm kiếm khớp chính xác bằng Elasticsearch"""
        processed_input = self.preprocess_text(user_input)
        
        query = {
            "query": {
                "match_phrase": {
                    "question": processed_input
                }
            }
        }
        
        try:
            response = self.es.search(index=self.index_name, body=query)
            if response['hits']['total']['value'] > 0:
                return response['hits']['hits'][0]['_source']['answer']
        except Exception as e:
            print(f"Lỗi tìm kiếm Elasticsearch: {e}")
        
        return None
    
    def get_response(self, user_input: str) -> str:
        """Lấy phản hồi cho câu hỏi của user với các phương pháp được cải thiện"""
        if not user_input.strip():
            return "Xin chào! Tôi có thể giúp gì cho bạn?"
        
        # 1. Thử exact match
        response = self.find_exact_match(user_input)
        if response:
            return response
        
        # 2. Thử partial string match (dễ match nhất)
        response = self.partial_string_match(user_input)
        if response:
            return response
        
        # 3. Thử keyword overlap match
        response = self.keyword_overlap_match(user_input)
        if response:
            return response
        
        # 4. Thử wildcard matching
        response = self.wildcard_match(user_input)
        if response:
            return response
        
        # 5. Thử fuzzy matching
        response = self.fuzzy_match(user_input)
        if response:
            return response
        
        # 6. Thử match phrase
        response = self.match_phrase(user_input)
        if response:
            return response
        
        # 7. Thử semantic search
        response = self.semantic_search(user_input)
        if response:
            return response
        
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn. Bạn có thể hỏi câu hỏi khác không?"
    
    def chat(self):
        """Bắt đầu cuộc trò chuyện"""
        print("=== RULE-BASED CHATBOT ===")
        if self.use_elasticsearch:
            print("🔍 Sử dụng Elasticsearch để tìm kiếm")
        else:
            print("📊 Sử dụng DataFrame để tìm kiếm")
        print("Chào bạn! Tôi là chatbot hỗ trợ trả lời câu hỏi.")
        print("Gõ 'quit', 'exit' hoặc 'bye' để thoát.")
        print("-" * 50)
        
        while True:
            user_input = input("\nBạn: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye', 'thoát']:
                print("Bot: Tạm biệt! Chúc bạn một ngày tốt lành!")
                break
            
            response = self.get_response(user_input)
            print(f"Bot: {response}")

def main():
    """Hàm chính để chạy chatbot"""
    try:
        chatbot = RuleBasedChatbot()
        chatbot.chat()
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file data.xlsx")
    except Exception as e:
        print(f"Lỗi: {e}")

# Khởi tạo Flask app
app = Flask(__name__)
chatbot = None

@app.route('/', methods=['POST'])
def chat_api():
    global chatbot
    try:
        # Khởi tạo chatbot nếu chưa có
        if chatbot is None:
            chatbot = RuleBasedChatbot(use_elasticsearch=False)
        
        data = request.get_json()
        user_input = data.get('chatInput', '')
        response = chatbot.get_response(user_input)
        return jsonify({'output': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        print("Khởi động API server tại http://localhost:8000")
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        main()
