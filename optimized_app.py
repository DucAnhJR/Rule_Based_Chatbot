"""
Optimized Rule-Based Chatbot - Focused on Accuracy
"""

import pandas as pd
import re
from typing import Optional
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata

class OptimizedChatbot:
    def __init__(self, data_file: str = 'data.xlsx'):
        """Khởi tạo chatbot tối ưu"""
        print("Loading data...")
        self.data = pd.read_excel(data_file)
        self.data = self.data.dropna(subset=['question', 'answer'])
        self.data['question'] = self.data['question'].astype(str)
        self.data['answer'] = self.data['answer'].astype(str)
        
        # Preprocessing đơn giản và hiệu quả
        self.data['processed_question'] = self.data['question'].apply(self.preprocess_text)
        
        # Sử dụng keyword có sẵn
        if 'keyword' in self.data.columns:
            self.data['keywords'] = self.data['keyword'].astype(str).fillna('')
        else:
            self.data['keywords'] = ''
        
        # Tạo TF-IDF đơn giản nhưng hiệu quả
        all_text = []
        for _, row in self.data.iterrows():
            combined = f"{row['processed_question']} {row['keywords']}"
            all_text.append(combined)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            lowercase=True
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_text)
        
        print(f"Loaded {len(self.data)} questions successfully!")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocessing đơn giản nhưng hiệu quả"""
        if not text:
            return ""
        
        text = str(text).lower().strip()
        text = unicodedata.normalize('NFC', text)
        
        # Xử lý từ viết tắt quan trọng
        text = re.sub(r'\bsv\b', 'sinh viên', text)
        text = re.sub(r'\bfptu\b', 'fpt', text)
        text = re.sub(r'\bđh\b', 'đại học', text)
        text = re.sub(r'\bbn\b', 'bao nhiêu', text)
        text = re.sub(r'\bk\b', 'không', text)
        text = re.sub(r'\bđc\b', 'được', text)
        text = re.sub(r'\bvs\b', 'với', text)
        text = re.sub(r'\bpt\b', 'phòng thi', text)
        
        # Làm sạch
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def exact_match(self, user_input: str) -> Optional[str]:
        """Exact matching"""
        user_processed = self.preprocess_text(user_input)
        
        for _, row in self.data.iterrows():
            if user_processed == row['processed_question']:
                return row['answer']
        return None
    
    def keyword_match(self, user_input: str) -> Optional[str]:
        """Keyword matching với scoring thông minh"""
        user_processed = self.preprocess_text(user_input)
        user_words = set(user_processed.split())
        
        # Từ khóa quan trọng với trọng số
        important_words = {
            'ngành': 10, 'học': 8, 'phí': 10, 'điểm': 8, 'thi': 8, 'phòng': 8,
            'sinh': 6, 'viên': 6, 'fpt': 8, 'môn': 8, 'bao': 6, 'nhiêu': 6,
            'mấy': 6, 'nào': 5, 'gì': 5, 'những': 5, 'các': 5, 'khi': 5,
            'ở': 5, 'đâu': 5, 'như': 4, 'thế': 4, 'sao': 4, 'làm': 4
        }
        
        best_match = None
        best_score = 0
        
        for _, row in self.data.iterrows():
            question_words = set(row['processed_question'].split())
            
            # Tính điểm
            score = 0
            
            # Exact word matches
            exact_matches = user_words.intersection(question_words)
            for word in exact_matches:
                if word in important_words:
                    score += important_words[word]
                else:
                    score += 3
            
            # Bonus cho coverage
            if len(user_words) > 0:
                coverage = len(exact_matches) / len(user_words)
                score += coverage * 15
            
            # Bonus cho keyword column
            if row['keywords']:
                keyword_words = set(row['keywords'].lower().split())
                keyword_matches = user_words.intersection(keyword_words)
                score += len(keyword_matches) * 12
            
            # Partial matches
            for user_word in user_words:
                if len(user_word) > 3:
                    for q_word in question_words:
                        if len(q_word) > 3 and (user_word in q_word or q_word in user_word):
                            score += 4
                            break
            
            if score > best_score and score > 10:  # Threshold
                best_score = score
                best_match = row['answer']
        
        return best_match
    
    def semantic_search(self, user_input: str) -> Optional[str]:
        """Semantic search tối ưu"""
        user_processed = self.preprocess_text(user_input)
        
        try:
            # Tạo vector cho input
            input_vector = self.tfidf_vectorizer.transform([user_processed])
            
            # Tính similarity
            similarities = cosine_similarity(input_vector, self.tfidf_matrix).flatten()
            
            # Tìm best match
            best_idx = np.argmax(similarities)
            best_similarity = similarities[best_idx]
            
            if best_similarity > 0.15:  # Threshold thấp để dễ match
                return self.data.iloc[best_idx]['answer']
            
            return None
        except:
            return None
    
    def phrase_match(self, user_input: str) -> Optional[str]:
        """Phrase matching - tìm cụm từ chung"""
        user_processed = self.preprocess_text(user_input)
        user_words = user_processed.split()
        
        best_match = None
        best_score = 0
        
        for _, row in self.data.iterrows():
            question_processed = row['processed_question']
            score = 0
            
            # Tìm bigrams và trigrams chung
            for i in range(len(user_words) - 1):
                bigram = ' '.join(user_words[i:i+2])
                if bigram in question_processed:
                    score += 15
            
            for i in range(len(user_words) - 2):
                trigram = ' '.join(user_words[i:i+3])
                if trigram in question_processed:
                    score += 25
            
            # Substring matching
            if len(user_processed) > 5 and user_processed in question_processed:
                score += 20
            
            if score > best_score and score > 10:
                best_score = score
                best_match = row['answer']
        
        return best_match
    
    def fuzzy_match(self, user_input: str) -> Optional[str]:
        """Fuzzy matching đơn giản"""
        user_processed = self.preprocess_text(user_input)
        user_words = set(user_processed.split())
        
        best_match = None
        best_similarity = 0
        
        for _, row in self.data.iterrows():
            question_words = set(row['processed_question'].split())
            
            if user_words and question_words:
                # Jaccard similarity
                intersection = len(user_words & question_words)
                union = len(user_words | question_words)
                jaccard = intersection / union if union > 0 else 0
                
                # Containment similarity
                containment = intersection / len(user_words) if len(user_words) > 0 else 0
                
                # Combined score
                similarity = (jaccard * 0.6) + (containment * 0.4)
                
                if similarity > best_similarity and similarity > 0.2:
                    best_similarity = similarity
                    best_match = row['answer']
        
        return best_match
    
    def get_response(self, user_input: str) -> str:
        """Lấy response với thứ tự tối ưu"""
        if not user_input.strip():
            return "Xin chào! Tôi có thể giúp gì cho bạn?"
        
        # Thứ tự ưu tiên: hiệu quả cao nhất trước
        
        # 1. Exact match
        result = self.exact_match(user_input)
        if result:
            return result
        
        # 2. Keyword match (hiệu quả nhất)
        result = self.keyword_match(user_input)
        if result:
            return result
        
        # 3. Phrase match
        result = self.phrase_match(user_input)
        if result:
            return result
        
        # 4. Semantic search
        result = self.semantic_search(user_input)
        if result:
            return result
        
        # 5. Fuzzy match (cuối cùng)
        result = self.fuzzy_match(user_input)
        if result:
            return result
        
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn. Bạn có thể hỏi câu hỏi khác không?"

# Flask app
app = Flask(__name__)
chatbot = None

@app.route('/', methods=['POST'])
def chat_api():
    global chatbot
    try:
        if chatbot is None:
            chatbot = OptimizedChatbot()
        
        data = request.get_json()
        user_input = data.get('chatInput', '')
        response = chatbot.get_response(user_input)
        return jsonify({'output': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    """Test function"""
    chatbot = OptimizedChatbot()
    
    test_questions = [
        "Những ngành học tại FPTU",
        "Sinh viên trễ bao nhiêu phút thì không được vào phòng thi?",
        "Đại học FPT có những ngành học nào?",
        "Các hành vi nào bị cấm trong phòng thi?",
        "Học phí của ngành công nghệ thông tin là bao nhiêu?"
    ]
    
    for q in test_questions:
        print(f"Q: {q}")
        print(f"A: {chatbot.get_response(q)[:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        print("Starting Optimized Chatbot API...")
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        main()
