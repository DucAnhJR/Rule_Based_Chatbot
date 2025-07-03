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
        """Tìm kiếm wildcard pattern - cải thiện với weighted scoring"""
        input_words = self.extract_important_words(user_input)
        
        best_match = None
        best_score = 0
        
        for idx, row in self.data.iterrows():
            question_words = self.extract_important_words(row['question'])
            question_text = self.preprocess_text(row['question'])
            
            score = 0
            matched_words = 0
            
            for input_word in input_words:
                word_matched = False
                
                # Exact match trong từ
                if input_word in question_words:
                    score += 10
                    word_matched = True
                
                # Partial match trong từ
                for question_word in question_words:
                    if len(input_word) > 1:
                        if input_word in question_word:
                            score += 5
                            word_matched = True
                        elif len(question_word) > 1 and question_word in input_word:
                            score += 3
                            word_matched = True
                
                # Match trong toàn bộ câu hỏi
                if len(input_word) > 2 and input_word in question_text:
                    score += 2
                    word_matched = True
                
                if word_matched:
                    matched_words += 1
            
            # Bonus cho tỷ lệ từ được match
            if len(input_words) > 0:
                match_ratio = matched_words / len(input_words)
                score += match_ratio * 10
            
            # Bonus cho độ dài từ khóa
            for word in input_words:
                if len(word) > 3:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = row['answer']
        
        return best_match if best_score > 3 else None
    
    def fuzzy_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm fuzzy matching - cải thiện với weighted scoring"""
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
                
                # Containment similarity (cả 2 chiều)
                containment1 = intersection / len(input_words) if len(input_words) > 0 else 0
                containment2 = intersection / len(question_words) if len(question_words) > 0 else 0
                
                # Weighted combination
                similarity = (jaccard * 0.4) + (containment1 * 0.4) + (containment2 * 0.2)
                
                # Bonus cho số từ khớp nhiều
                if intersection > 1:
                    similarity += intersection * 0.1
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = row['answer']
        
        # Ngưỡng thấp hơn để dễ match hơn
        return best_match if best_similarity > 0.05 else None
    
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
    
    def advanced_similarity_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm dựa trên similarity nâng cao - tối ưu tốc độ"""
        input_words = self.extract_important_words(user_input)
        processed_input = self.preprocess_text(user_input)
        
        # Giới hạn số từ để tăng tốc độ
        if len(input_words) > 10:
            input_words = input_words[:10]
        
        best_match = None
        best_score = 0
        
        for idx, row in self.data.iterrows():
            question_words = self.extract_important_words(row['question'])
            processed_question = self.preprocess_text(row['question'])
            
            total_score = 0
            
            # 1. Exact word matching (nhanh)
            input_set = set(input_words)
            question_set = set(question_words)
            if input_set and question_set:
                exact_matches = len(input_set & question_set)
                total_score += exact_matches * 10
                
                # Early exit nếu có nhiều exact matches
                if exact_matches >= len(input_words) * 0.7:
                    return row['answer']
            
            # 2. Partial word matching (giới hạn)
            partial_count = 0
            for input_word in input_words[:5]:  # Chỉ kiểm tra 5 từ đầu
                for question_word in question_words[:10]:  # Chỉ kiểm tra 10 từ đầu
                    if len(input_word) > 2 and len(question_word) > 2:
                        if input_word in question_word or question_word in input_word:
                            total_score += 5
                            partial_count += 1
                            break
                if partial_count >= 3:  # Đủ rồi, không cần kiểm tra thêm
                    break
            
            # 3. Sequence matching (đơn giản hóa)
            if len(processed_input) > 5 and processed_input in processed_question:
                total_score += 15
            
            if total_score > best_score:
                best_score = total_score
                best_match = row['answer']
        
        return best_match if best_score > 8 else None
    
    def levenshtein_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm dựa trên khoảng cách Levenshtein đơn giản - tối ưu tốc độ"""
        processed_input = self.preprocess_text(user_input)
        
        # Giới hạn độ dài để tăng tốc độ
        if len(processed_input) > 50:
            processed_input = processed_input[:50]
        
        def simple_distance(s1, s2):
            # Tối ưu: kiểm tra độ dài trước
            len_diff = abs(len(s1) - len(s2))
            if len_diff > min(len(s1), len(s2)) * 0.7:  # Quá khác biệt
                return float('inf')
            
            if len(s1) == 0:
                return len(s2)
            if len(s2) == 0:
                return len(s1)
            
            # Ma trận DP đơn giản với giới hạn
            if len(s1) > 30 or len(s2) > 30:  # Quá dài, bỏ qua
                return float('inf')
            
            prev_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                curr_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev_row[j + 1] + 1
                    deletions = curr_row[j] + 1
                    substitutions = prev_row[j] + (c1 != c2)
                    curr_row.append(min(insertions, deletions, substitutions))
                prev_row = curr_row
            
            return prev_row[-1]
        
        best_match = None
        min_distance = float('inf')
        
        for idx, row in self.data.iterrows():
            processed_question = self.preprocess_text(row['question'])
            
            # Tối ưu: kiểm tra nhanh trước
            if abs(len(processed_input) - len(processed_question)) > 20:
                continue
            
            if processed_input and processed_question:
                distance = simple_distance(processed_input[:30], processed_question[:30])  # Giới hạn độ dài
                max_len = max(len(processed_input), len(processed_question))
                
                if max_len > 0 and distance != float('inf'):
                    similarity = 1 - (distance / max_len)
                    if similarity > 0.5 and distance < min_distance:  # Tăng ngưỡng
                        min_distance = distance
                        best_match = row['answer']
        
        return best_match
    
    def flexible_keyword_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm từ khóa linh hoạt với scoring"""
        input_words = [word.lower() for word in self.extract_important_words(user_input) if len(word) > 1]
        
        if not input_words:
            return None
        
        scored_matches = []
        
        for idx, row in self.data.iterrows():
            question_words = [word.lower() for word in self.extract_important_words(row['question']) if len(word) > 1]
            question_text = self.preprocess_text(row['question'])
            
            score = 0
            
            # Điểm cho exact match
            for input_word in input_words:
                if input_word in question_words:
                    score += 10
                
                # Điểm cho partial match
                for question_word in question_words:
                    if len(input_word) > 2 and input_word in question_word:
                        score += 5
                    elif len(question_word) > 2 and question_word in input_word:
                        score += 5
                
                # Điểm cho substring trong câu hỏi
                if input_word in question_text:
                    score += 3
            
            # Bonus cho nhiều từ match
            matched_words = set(input_words) & set(question_words)
            if len(matched_words) > 1:
                score += len(matched_words) * 5
            
            if score > 0:
                scored_matches.append((score, row['answer']))
        
        if scored_matches:
            scored_matches.sort(reverse=True)
            return scored_matches[0][1]
        
        return None
    
    def ngram_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm dựa trên n-gram"""
        def get_ngrams(text, n):
            words = text.split()
            return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        processed_input = self.preprocess_text(user_input)
        input_bigrams = set(get_ngrams(processed_input, 2))
        input_trigrams = set(get_ngrams(processed_input, 3))
        
        best_match = None
        best_score = 0
        
        for idx, row in self.data.iterrows():
            processed_question = self.preprocess_text(row['question'])
            question_bigrams = set(get_ngrams(processed_question, 2))
            question_trigrams = set(get_ngrams(processed_question, 3))
            
            score = 0
            
            # Trigram matches (cao điểm nhất)
            trigram_matches = len(input_trigrams & question_trigrams)
            score += trigram_matches * 15
            
            # Bigram matches
            bigram_matches = len(input_bigrams & question_bigrams)
            score += bigram_matches * 8
            
            if score > best_score:
                best_score = score
                best_match = row['answer']
        
        return best_match if best_score > 10 else None
    
    def contextual_match(self, user_input: str) -> Optional[str]:
        """Tìm kiếm theo ngữ cảnh"""
        # Các từ chỉ thị quan trọng
        question_indicators = ['gì', 'sao', 'nào', 'đâu', 'khi', 'như', 'thế', 'làm', 'có', 'được']
        
        processed_input = self.preprocess_text(user_input)
        input_words = processed_input.split()
        
        best_match = None
        best_relevance = 0
        
        for idx, row in self.data.iterrows():
            processed_question = self.preprocess_text(row['question'])
            question_words = processed_question.split()
            
            relevance = 0
            
            # Kiểm tra từ chỉ thị
            for indicator in question_indicators:
                if indicator in input_words and indicator in question_words:
                    relevance += 5
            
            # Kiểm tra từ khóa trong ngữ cảnh
            for i, word in enumerate(input_words):
                if word in question_words:
                    # Bonus cho vị trí tương tự
                    try:
                        question_index = question_words.index(word)
                        position_similarity = 1 - abs(i - question_index) / max(len(input_words), len(question_words))
                        relevance += position_similarity * 3
                    except:
                        relevance += 2
            
            if relevance > best_relevance:
                best_relevance = relevance
                best_match = row['answer']
        
        return best_match if best_relevance > 3 else None
    
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
        """Lấy phản hồi cho câu hỏi của user với hệ thống tối ưu tốc độ"""
        if not user_input.strip():
            return "Xin chào! Tôi có thể giúp gì cho bạn?"
        
        # Thứ tự tối ưu: từ nhanh nhất đến chậm nhất
        
        # 1. Exact match (nhanh nhất)
        exact_result = self.find_exact_match(user_input)
        if exact_result:
            return exact_result
        
        # 2. Partial string match (rất nhanh)
        partial_result = self.partial_string_match(user_input)
        if partial_result:
            return partial_result
        
        # 3. Flexible keyword match (nhanh, hiệu quả cao)
        flex_keyword_result = self.flexible_keyword_match(user_input)
        if flex_keyword_result:
            return flex_keyword_result
        
        # 4. Wildcard matching (nhanh)
        wildcard_result = self.wildcard_match(user_input)
        if wildcard_result:
            return wildcard_result
        
        # 5. N-gram match (trung bình)
        ngram_result = self.ngram_match(user_input)
        if ngram_result:
            return ngram_result
        
        # 6. Fuzzy matching (trung bình)
        fuzzy_result = self.fuzzy_match(user_input)
        if fuzzy_result:
            return fuzzy_result
        
        # 7. Contextual match (trung bình)
        context_result = self.contextual_match(user_input)
        if context_result:
            return context_result
        
        # 8. Semantic search (nhanh với TF-IDF)
        semantic_result = self.semantic_search(user_input)
        if semantic_result:
            return semantic_result
        
        # 9. Advanced similarity (chậm hơn - chỉ khi cần thiết)
        adv_sim_result = self.advanced_similarity_match(user_input)
        if adv_sim_result:
            return adv_sim_result
        
        # 10. Levenshtein (chậm nhất - cuối cùng)
        leven_result = self.levenshtein_match(user_input)
        if leven_result:
            return leven_result
        
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
