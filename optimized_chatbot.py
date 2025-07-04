"""
Enhanced Rule-Based Chatbot - Optimized for 65% Accuracy
"""

import pandas as pd
import re
from typing import Optional, List, Tuple, Dict
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata
from collections import Counter

class OptimizedChatbot:
    def __init__(self, data_file: str = 'data.xlsx'):
        """Khởi tạo chatbot tối ưu cho accuracy 65%"""
        print("Loading data...")
        self.data = pd.read_excel(data_file)
        self.data = self.data.dropna(subset=['question', 'answer'])
        self.data['question'] = self.data['question'].astype(str)
        self.data['answer'] = self.data['answer'].astype(str)
        
        # Preprocessing tối ưu
        self.data['processed_question'] = self.data['question'].apply(self.preprocess_text)
        self.data['normalized_question'] = self.data['question'].apply(self.normalize_question)
        
        # Keyword processing
        if 'keyword' in self.data.columns:
            self.data['keywords'] = self.data['keyword'].astype(str).fillna('')
        else:
            self.data['keywords'] = ''
        
        # Tạo word frequency cho rare word detection
        self._build_word_frequency()
        
        # Tạo TF-IDF matrices
        self._create_tfidf_matrices()
        
        print(f"Loaded {len(self.data)} questions successfully!")
    
    def _build_word_frequency(self):
        """Tạo word frequency dictionary để xác định rare words"""
        all_words = []
        for _, row in self.data.iterrows():
            all_words.extend(row['processed_question'].split())
        self.word_freq = Counter(all_words)
        
        # Xác định rare words (xuất hiện ít hơn 3 lần)
        self.rare_words = set(word for word, freq in self.word_freq.items() if freq < 3 and len(word) > 4)
    
    def _create_tfidf_matrices(self):
        """Tạo TF-IDF matrices tối ưu"""
        
        # Matrix 1: Combined (processed + keywords) - high weight
        combined_text = []
        for _, row in self.data.iterrows():
            text = f"{row['processed_question']} {row['keywords']}"
            combined_text.append(text)
        
        self.tfidf_vectorizer1 = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.85,
            lowercase=True,
            stop_words=None
        )
        self.tfidf_matrix1 = self.tfidf_vectorizer1.fit_transform(combined_text)
        
        # Matrix 2: Normalized questions - medium weight
        self.tfidf_vectorizer2 = TfidfVectorizer(
            max_features=6000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.9,
            lowercase=True
        )
        self.tfidf_matrix2 = self.tfidf_vectorizer2.fit_transform(self.data['normalized_question'].tolist())
        
        # Matrix 3: Original questions for phrase matching - low weight
        self.tfidf_vectorizer3 = TfidfVectorizer(
            max_features=4000,
            ngram_range=(2, 3),
            min_df=1,
            max_df=0.95,
            lowercase=True
        )
        self.tfidf_matrix3 = self.tfidf_vectorizer3.fit_transform(self.data['question'].str.lower().tolist())
    
    def preprocess_text(self, text: str) -> str:
        """Preprocessing nhẹ nhàng hơn để giữ lại thông tin"""
        if not text:
            return ""
        
        text = str(text).lower().strip()
        text = unicodedata.normalize('NFC', text)
        
        # Core normalizations only
        normalizations = {
            # Essential abbreviations
            r'\b(sv|sinh viên)\b': 'sinh viên',
            r'\b(fptu?|đại học fpt|trường fpt)\b': 'fpt',
            r'\b(đh|đại học)\b': 'đại học',
            r'\b(hp|học phí)\b': 'học phí',
            r'\b(bn|bao nhiêu)\b': 'bao nhiêu',
            r'\b(tg|thời gian)\b': 'thời gian',
            r'\b(pt|phòng thi)\b': 'phòng thi',
            
            # Essential synonyms
            r'\b(chi phí|phí tổn)\b': 'phí',
            r'\b(chuyên ngành|ngành đào tạo)\b': 'ngành',
            r'\b(kiểm tra|bài thi)\b': 'thi',
            r'\b(học sinh)\b': 'sinh viên',
            r'\b(giáo viên|thầy cô|giảng viên)\b': 'giảng viên',
            
            # Question words
            r'\b(như thế nào|làm sao|cách nào)\b': 'như thế nào',
            r'\b(ở đâu|tại đâu|chỗ nào)\b': 'ở đâu',
            r'\b(khi nào|lúc nào)\b': 'khi nào',
        }
        
        for pattern, replacement in normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Light cleanup
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def normalize_question(self, text: str) -> str:
        """Normalize nhẹ cho semantic matching"""
        text = self.preprocess_text(text)
        
        # Remove common stop words only
        stop_words = ['thì', 'mà', 'rồi', 'ạ', 'ơi', 'nhé', 'à']
        words = text.split()
        words = [word for word in words if word not in stop_words and len(word) > 1]
        
        return ' '.join(words)
    
    def semantic_search(self, user_input: str, k: int = 12) -> List[Tuple[int, float]]:
        """Semantic search tối ưu với multiple strategies"""
        candidates = []
        
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        
        # Strategy 1: Combined text (cao nhất)
        try:
            input_vector1 = self.tfidf_vectorizer1.transform([user_processed])
            similarities1 = cosine_similarity(input_vector1, self.tfidf_matrix1).flatten()
            top_indices1 = np.argsort(similarities1)[-k*2:][::-1]
            for idx in top_indices1:
                if similarities1[idx] > 0.05:  # Giảm threshold
                    candidates.append((idx, similarities1[idx] * 1.3, 'combined'))
        except:
            pass
        
        # Strategy 2: Normalized questions
        try:
            input_vector2 = self.tfidf_vectorizer2.transform([user_normalized])
            similarities2 = cosine_similarity(input_vector2, self.tfidf_matrix2).flatten()
            top_indices2 = np.argsort(similarities2)[-k*2:][::-1]
            for idx in top_indices2:
                if similarities2[idx] > 0.05:
                    candidates.append((idx, similarities2[idx] * 1.0, 'normalized'))
        except:
            pass
        
        # Strategy 3: Original questions
        try:
            input_vector3 = self.tfidf_vectorizer3.transform([user_input.lower()])
            similarities3 = cosine_similarity(input_vector3, self.tfidf_matrix3).flatten()
            top_indices3 = np.argsort(similarities3)[-k*2:][::-1]
            for idx in top_indices3:
                if similarities3[idx] > 0.03:
                    candidates.append((idx, similarities3[idx] * 0.8, 'original'))
        except:
            pass
        
        # Aggregate scores
        if candidates:
            score_dict = {}
            for idx, score, strategy in candidates:
                if idx not in score_dict:
                    score_dict[idx] = {'scores': [], 'strategies': []}
                score_dict[idx]['scores'].append(score)
                score_dict[idx]['strategies'].append(strategy)
            
            # Calculate final scores
            final_candidates = []
            for idx, data in score_dict.items():
                scores = data['scores']
                strategies = data['strategies']
                
                # Weighted average với bonus cho multiple strategies
                final_score = max(scores) * 0.7 + (sum(scores) / len(scores)) * 0.3
                
                # Strategy diversity bonus
                if len(set(strategies)) > 1:
                    final_score *= 1.15
                
                final_candidates.append((idx, final_score))
            
            # Sort and return
            final_candidates.sort(key=lambda x: x[1], reverse=True)
            return final_candidates[:k]
        
        return []
    
    def calculate_keyword_score(self, user_input: str, row) -> float:
        """Simplified but effective keyword scoring"""
        user_processed = self.preprocess_text(user_input)
        user_words = set(user_processed.split())
        
        question_words = set(row['processed_question'].split())
        
        # Basic overlap
        overlap = user_words.intersection(question_words)
        if not overlap:
            return 0.0
        
        # Keyword importance weights
        important_keywords = {
            'ngành': 30, 'học': 25, 'phí': 30, 'điểm': 25, 'thi': 25, 'phòng': 20,
            'sinh': 15, 'viên': 15, 'fpt': 25, 'môn': 20, 'bao': 15, 'nhiêu': 15,
            'đại': 15, 'trường': 15, 'thời': 15, 'gian': 15, 'như': 10, 'thế': 10,
            'nào': 10, 'ở': 10, 'đâu': 10, 'khi': 10, 'cách': 12, 'làm': 10
        }
        
        # Calculate weighted overlap score
        weighted_score = 0
        for word in overlap:
            if word in important_keywords:
                weighted_score += important_keywords[word]
            elif word in self.rare_words:
                weighted_score += 20  # Bonus for rare words
            else:
                weighted_score += 8   # Base score
        
        # Coverage bonus
        user_coverage = len(overlap) / len(user_words) if user_words else 0
        question_coverage = len(overlap) / len(question_words) if question_words else 0
        
        coverage_bonus = (user_coverage + question_coverage) * 15
        
        # Keyword column bonus
        keyword_bonus = 0
        if row['keywords']:
            keyword_words = set(str(row['keywords']).lower().split())
            keyword_overlap = user_words.intersection(keyword_words)
            keyword_bonus = len(keyword_overlap) * 12
        
        # Total score
        total_score = weighted_score + coverage_bonus + keyword_bonus
        
        return total_score
    
    def exact_match(self, user_input: str) -> Optional[str]:
        """Exact matching optimized"""
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        
        # Check against processed and normalized versions
        for _, row in self.data.iterrows():
            if (user_processed == row['processed_question'] or
                user_normalized == row['normalized_question']):
                return row['answer']
        
        return None
    
    def hybrid_search(self, user_input: str) -> Optional[str]:
        """Main hybrid search method"""
        # Get semantic candidates
        semantic_candidates = self.semantic_search(user_input, k=15)
        
        if not semantic_candidates:
            return None
        
        # Re-rank với combined scoring
        final_candidates = []
        
        for idx, semantic_score in semantic_candidates:
            row = self.data.iloc[idx]
            
            # Calculate component scores
            keyword_score = self.calculate_keyword_score(user_input, row)
            fuzzy_score = self._calculate_fuzzy_score(user_input, row)
            length_score = self._calculate_length_score(user_input, row)
            
            # Weighted combination - balanced approach
            final_score = (
                semantic_score * 0.4 +        # 40% semantic
                keyword_score * 0.4 +         # 40% keyword
                fuzzy_score * 0.15 +          # 15% fuzzy
                length_score * 0.05           # 5% length
            )
            
            # Boosting logic
            if semantic_score > 0.5 and keyword_score > 40:
                final_score *= 1.2
            elif semantic_score > 0.3 and keyword_score > 30:
                final_score *= 1.1
            
            final_candidates.append((final_score, row['answer'], idx))
        
        # Return best match
        if final_candidates:
            final_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_answer, best_idx = final_candidates[0]
            
            # Dynamic threshold
            if len(final_candidates) > 1:
                second_score = final_candidates[1][0]
                if best_score > second_score * 1.3:
                    threshold = 0.1
                else:
                    threshold = 0.15
            else:
                threshold = 0.12
            
            if best_score > threshold:
                return best_answer
        
        return None
    
    def _calculate_fuzzy_score(self, user_input: str, row) -> float:
        """Simplified fuzzy scoring"""
        user_words = set(self.preprocess_text(user_input).split())
        question_words = set(row['processed_question'].split())
        
        if not user_words or not question_words:
            return 0.0
        
        intersection = len(user_words & question_words)
        union = len(user_words | question_words)
        
        jaccard = intersection / union if union > 0 else 0
        containment = intersection / len(user_words) if user_words else 0
        
        return (jaccard * 0.6 + containment * 0.4)
    
    def _calculate_length_score(self, user_input: str, row) -> float:
        """Length similarity score"""
        user_len = len(user_input.split())
        question_len = len(row['question'].split())
        
        if user_len == 0 or question_len == 0:
            return 0.0
        
        ratio = min(user_len, question_len) / max(user_len, question_len)
        return ratio
    
    def advanced_keyword_match(self, user_input: str) -> Optional[str]:
        """Advanced keyword matching as fallback"""
        user_words = set(self.preprocess_text(user_input).split())
        
        best_match = None
        best_score = 0
        
        for _, row in self.data.iterrows():
            score = self.calculate_keyword_score(user_input, row)
            
            if score > best_score and score > 25:  # Lower threshold
                best_score = score
                best_match = row['answer']
        
        return best_match
    
    def get_response(self, user_input: str) -> str:
        """Main response method với optimized flow"""
        if not user_input.strip():
            return "Xin chào! Tôi có thể giúp gì cho bạn?"
        
        # Method 1: Exact match
        try:
            result = self.exact_match(user_input)
            if result:
                return result
        except Exception as e:
            print(f"Error in exact_match: {e}")
        
        # Method 2: Hybrid search (main method)
        try:
            result = self.hybrid_search(user_input)
            if result:
                return result
        except Exception as e:
            print(f"Error in hybrid_search: {e}")
        
        # Method 3: Advanced keyword matching (fallback)
        try:
            result = self.advanced_keyword_match(user_input)
            if result:
                return result
        except Exception as e:
            print(f"Error in advanced_keyword_match: {e}")
        
        # Method 4: Semantic-only search (last resort)
        try:
            semantic_candidates = self.semantic_search(user_input, k=5)
            if semantic_candidates:
                best_idx, best_score = semantic_candidates[0]
                if best_score > 0.08:  # Very low threshold
                    return self.data.iloc[best_idx]['answer']
        except Exception as e:
            print(f"Error in semantic fallback: {e}")
        
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
        response = chatbot.get_response(q)
        print(f"A: {response[:100]}...")
        print("-" * 50)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        print("Starting Optimized Chatbot API...")
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        main()
