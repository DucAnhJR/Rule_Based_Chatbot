"""
Advanced Rule-Based Chatbot - Top-K Semantic + Normalized Scoring
"""

import pandas as pd
import re
from typing import Optional, List, Tuple
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata

class AdvancedChatbot:
    def __init__(self, data_file: str = 'data.xlsx'):
        """Khởi tạo chatbot nâng cao với top-k semantic"""
        print("Loading data...")
        self.data = pd.read_excel(data_file)
        self.data = self.data.dropna(subset=['question', 'answer'])
        self.data['question'] = self.data['question'].astype(str)
        self.data['answer'] = self.data['answer'].astype(str)
        
        # Preprocessing nâng cao
        self.data['processed_question'] = self.data['question'].apply(self.preprocess_text)
        self.data['normalized_question'] = self.data['question'].apply(self.normalize_question)
        
        # Sử dụng keyword có sẵn
        if 'keyword' in self.data.columns:
            self.data['keywords'] = self.data['keyword'].astype(str).fillna('')
        else:
            self.data['keywords'] = ''
        
        # Tạo multiple TF-IDF matrices cho different strategies
        self._create_tfidf_matrices()
        
        print(f"Loaded {len(self.data)} questions successfully!")
    
    def _create_tfidf_matrices(self):
        """Tạo nhiều TF-IDF matrix cho các chiến lược khác nhau"""
        
        # Matrix 1: Processed questions + keywords
        combined_text1 = []
        for _, row in self.data.iterrows():
            text = f"{row['processed_question']} {row['keywords']}"
            combined_text1.append(text)
        
        self.tfidf_vectorizer1 = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),  # Tăng lên 3-gram
            min_df=1,
            max_df=0.7,
            lowercase=True
        )
        self.tfidf_matrix1 = self.tfidf_vectorizer1.fit_transform(combined_text1)
        
        # Matrix 2: Normalized questions only
        self.tfidf_vectorizer2 = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            lowercase=True
        )
        self.tfidf_matrix2 = self.tfidf_vectorizer2.fit_transform(self.data['normalized_question'].tolist())
        
        # Matrix 3: Original questions
        self.tfidf_vectorizer3 = TfidfVectorizer(
            max_features=6000,
            ngram_range=(2, 4),  # Focus on phrases
            min_df=1,
            max_df=0.9,
            lowercase=True
        )
        self.tfidf_matrix3 = self.tfidf_vectorizer3.fit_transform(self.data['question'].str.lower().tolist())
    
    def normalize_question(self, text: str) -> str:
        """Enhanced normalize câu hỏi để tăng khả năng match"""
        text = self.preprocess_text(text)
        
        # Chuẩn hóa câu hỏi patterns
        question_normalizations = {
            # Question words normalization
            r'\b(những|các)\s+': '',
            r'\b(tại|ở)\s+(fptu?|đại học fpt|trường fpt)\b': 'fpt',
            r'\b(fptu?|đại học fpt|trường fpt)\s+(có|tại)\b': 'fpt',
            r'\b(bao nhiêu|mấy)\s+(phút|giờ|thời gian)\b': 'bao nhiêu thời gian',
            r'\b(bao nhiêu|mấy)\s+(tiền|phí)\b': 'bao nhiêu phí',
            r'\b(nào|gì)\s*$': '',  # Remove trailing question words
            r'\b(như thế nào|làm sao|cách nào|ra sao)\b': 'như thế nào',
            r'\b(ở đâu|tại đâu|chỗ nào|nơi nào)\b': 'ở đâu',
            r'\b(khi nào|lúc nào|thời gian nào)\b': 'khi nào',
            
            # Content normalization
            r'\b(sinh viên|học sinh|sv)\b': 'sinh viên',
            r'\b(học phí|chi phí học|phí học)\b': 'học phí',
            r'\b(ngành học|chuyên ngành|ngành đào tạo)\b': 'ngành học',
            r'\b(phòng thi|phòng kiểm tra)\b': 'phòng thi',
            r'\b(điểm thi|kết quả thi)\b': 'điểm thi',
            r'\b(đại học fpt|trường fpt|fpt university)\b': 'fpt',
            
            # Time expressions
            r'\b(thời gian|giờ giấc|lịch trình)\b': 'thời gian',
            r'\b(phút|giờ)\b': 'thời gian',
            
            # Common phrases
            r'\b(có được|được không|có thể)\b': 'được',
            r'\b(cần phải|phải|cần)\b': 'cần',
            r'\b(và|với|cùng)\b': 'và',
        }
        
        # Apply normalizations
        for pattern, replacement in question_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Remove redundant words
        redundant_words = ['thì', 'mà', 'rồi', 'đây', 'kia', 'này', 'đó', 'vậy', 'ạ', 'ơi']
        words = text.split()
        words = [word for word in words if word not in redundant_words]
        text = ' '.join(words)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced preprocessing với better synonym handling"""
        if not text:
            return ""
        
        text = str(text).lower().strip()
        text = unicodedata.normalize('NFC', text)
        
        # Enhanced synonym and abbreviation handling
        text_normalizations = {
            # Abbreviations
            r'\b(sv|sinh viên)\b': 'sinh viên',
            r'\b(fptu?|đại học fpt|trường fpt|fpt university)\b': 'fpt',
            r'\b(đh|đại học)\b': 'đại học',
            r'\b(bn|bao nhiêu)\b': 'bao nhiêu',
            r'\b(k|ko|không)\b': 'không',
            r'\b(đc|được)\b': 'được',
            r'\b(vs|với)\b': 'với',
            r'\b(pt|phòng thi)\b': 'phòng thi',
            r'\b(mấy|bao nhiêu)\b': 'bao nhiêu',
            r'\b(tg|thời gian)\b': 'thời gian',
            r'\b(hp|học phí)\b': 'học phí',
            r'\b(hs|học sinh)\b': 'sinh viên',
            
            # Question words standardization
            r'\b(như thế nào|làm sao|cách nào|ra sao)\b': 'như thế nào',
            r'\b(ở đâu|tại đâu|chỗ nào|nơi nào)\b': 'ở đâu',
            r'\b(khi nào|lúc nào|thời gian nào)\b': 'khi nào',
            r'\b(bao lâu|mất bao lâu)\b': 'bao lâu',
            r'\b(có được|được không|có thể)\b': 'được',
            
            # Content synonyms
            r'\b(chi phí|phí tổn|giá cả)\b': 'phí',
            r'\b(chuyên ngành|ngành đào tạo)\b': 'ngành',
            r'\b(kiểm tra|bài thi)\b': 'thi',
            r'\b(học sinh|sinh viên)\b': 'sinh viên',
            r'\b(giáo viên|thầy cô|giảng viên)\b': 'giảng viên',
            r'\b(phòng học|lớp học)\b': 'phòng',
            r'\b(thời khóa biểu|lịch học)\b': 'lịch',
            
            # Units and measurements
            r'\b(triệu|tr)\b': 'triệu',
            r'\b(nghìn|k)\b': 'nghìn',
            r'\b(giờ|h)\b': 'giờ',
            r'\b(phút|ph)\b': 'phút',
        }
        
        # Apply normalizations
        for pattern, replacement in text_normalizations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean punctuation and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def top_k_semantic_search(self, user_input: str, k: int = 8) -> List[Tuple[int, float]]:
        """Top-K semantic search với multiple strategies - tăng k để có nhiều candidate hơn"""
        candidates = []
        
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        
        # Strategy 1: TF-IDF Matrix 1 (processed + keywords)
        try:
            input_vector1 = self.tfidf_vectorizer1.transform([user_processed])
            similarities1 = cosine_similarity(input_vector1, self.tfidf_matrix1).flatten()
            top_indices1 = np.argsort(similarities1)[-k*2:][::-1]  # Lấy nhiều hơn để rerank
            for idx in top_indices1:
                if similarities1[idx] > 0.08:  # Giảm threshold
                    candidates.append((idx, similarities1[idx] * 1.0, 'tfidf1'))
        except:
            pass
        
        # Strategy 2: TF-IDF Matrix 2 (normalized)
        try:
            input_vector2 = self.tfidf_vectorizer2.transform([user_normalized])
            similarities2 = cosine_similarity(input_vector2, self.tfidf_matrix2).flatten()
            top_indices2 = np.argsort(similarities2)[-k*2:][::-1]
            for idx in top_indices2:
                if similarities2[idx] > 0.08:
                    candidates.append((idx, similarities2[idx] * 1.2, 'tfidf2'))
        except:
            pass
        
        # Strategy 3: TF-IDF Matrix 3 (phrases)
        try:
            input_vector3 = self.tfidf_vectorizer3.transform([user_input.lower()])
            similarities3 = cosine_similarity(input_vector3, self.tfidf_matrix3).flatten()
            top_indices3 = np.argsort(similarities3)[-k*2:][::-1]
            for idx in top_indices3:
                if similarities3[idx] > 0.04:
                    candidates.append((idx, similarities3[idx] * 0.8, 'tfidf3'))
        except:
            pass
        
        # Normalize scores và aggregate
        if candidates:
            # Group by index
            score_dict = {}
            for idx, score, strategy in candidates:
                if idx not in score_dict:
                    score_dict[idx] = []
                score_dict[idx].append(score)
            
            # Calculate final scores
            final_candidates = []
            for idx, scores in score_dict.items():
                # Use max + average boost + strategy diversity bonus
                final_score = max(scores) + (sum(scores) / len(scores)) * 0.3
                if len(scores) > 1:  # Bonus for multiple strategies
                    final_score += 0.1
                final_candidates.append((idx, final_score))
            
            # Sort and return top-k
            final_candidates.sort(key=lambda x: x[1], reverse=True)
            return final_candidates[:k]
        
        return []
    
    def exact_match(self, user_input: str) -> Optional[str]:
        """Exact matching với nhiều variants"""
        variants = [
            self.preprocess_text(user_input),
            self.normalize_question(user_input),
            user_input.lower().strip()
        ]
        
        for variant in variants:
            for _, row in self.data.iterrows():
                question_variants = [
                    row['processed_question'],
                    row['normalized_question'],
                    row['question'].lower().strip()
                ]
                if variant in question_variants:
                    return row['answer']
        return None
    
    def advanced_keyword_match(self, user_input: str) -> Optional[str]:
        """Enhanced keyword matching với advanced overlap scoring"""
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = set(user_processed.split())
        user_norm_words = set(user_normalized.split())
        
        scored_matches = []
        
        for idx, row in self.data.iterrows():
            # Use the new keyword overlap scoring function
            keyword_overlap_score = self.calculate_keyword_overlap_score(user_input, row)
            
            # Additional scoring components
            additional_score = 0
            
            # Context-aware matching
            question_words = set(row['processed_question'].split())
            
            # Question pattern matching
            question_patterns = {
                'what': ['gì', 'nào', 'những', 'các'],
                'how_many': ['bao', 'nhiêu', 'mấy'],
                'when': ['khi', 'lúc', 'thời'],
                'where': ['ở', 'đâu', 'chỗ'],
                'how': ['như', 'thế', 'nào', 'cách', 'làm']
            }
            
            user_pattern = None
            row_pattern = None
            
            for pattern, words in question_patterns.items():
                if any(word in user_words for word in words):
                    user_pattern = pattern
                if any(word in question_words for word in words):
                    row_pattern = pattern
            
            if user_pattern and user_pattern == row_pattern:
                additional_score += 20
            
            # Semantic field matching
            semantic_fields = {
                'education': ['ngành', 'học', 'môn', 'đại', 'trường', 'giáo', 'dục', 'sinh', 'viên'],
                'finance': ['phí', 'tiền', 'chi', 'giá', 'bao', 'nhiêu', 'triệu', 'nghìn'],
                'examination': ['thi', 'điểm', 'phòng', 'kiểm', 'tra', 'cấm', 'trễ'],
                'time': ['thời', 'gian', 'giờ', 'phút', 'khi', 'lúc'],
                'location': ['ở', 'đâu', 'chỗ', 'nơi', 'phòng', 'địa']
            }
            
            user_fields = set()
            row_fields = set()
            
            for field, keywords in semantic_fields.items():
                if any(keyword in user_words for keyword in keywords):
                    user_fields.add(field)
                if any(keyword in question_words for keyword in keywords):
                    row_fields.add(field)
            
            field_overlap = len(user_fields.intersection(row_fields))
            additional_score += field_overlap * 12
            
            # Length and structure similarity
            len_ratio = min(len(user_input), len(row['question'])) / max(len(user_input), len(row['question']))
            additional_score += len_ratio * 15
            
            # Word count similarity
            word_count_ratio = min(len(user_words), len(question_words)) / max(len(user_words), len(question_words))
            additional_score += word_count_ratio * 10
            
            # Combine scores
            total_score = keyword_overlap_score + additional_score
            
            if total_score > 25:  # Threshold
                scored_matches.append((total_score, row['answer'], idx))
        
        # Return best match với adaptive threshold
        if scored_matches:
            scored_matches.sort(key=lambda x: x[0], reverse=True)
            
            top_score = scored_matches[0][0]
            
            # Adaptive threshold based on score range
            if len(scored_matches) > 1:
                second_score = scored_matches[1][0]
                score_gap = top_score - second_score
                
                if score_gap > 30:  # Large gap
                    threshold_ratio = 0.3
                elif score_gap > 15:  # Medium gap
                    threshold_ratio = 0.5
                else:  # Small gap
                    threshold_ratio = 0.7
            else:
                threshold_ratio = 0.4
            
            max_possible_score = 150  # Estimate based on scoring system
            threshold = max_possible_score * threshold_ratio
            
            if top_score > threshold:
                return scored_matches[0][1]
        
        return None
    
    def calculate_keyword_overlap_score(self, user_input: str, row) -> float:
        """Calculate advanced keyword overlap score"""
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = set(user_processed.split())
        user_norm_words = set(user_normalized.split())
        
        # Get question words
        question_words = set(row['processed_question'].split())
        norm_question_words = set(row['normalized_question'].split())
        
        # Advanced keyword importance với context
        high_importance_keywords = {
            'ngành': 25, 'học': 20, 'phí': 25, 'điểm': 22, 'thi': 22, 'phòng': 18,
            'sinh': 15, 'viên': 15, 'fpt': 22, 'môn': 18, 'bao': 15, 'nhiêu': 15,
            'đại': 15, 'trường': 12, 'thời': 15, 'gian': 15, 'cách': 12, 'làm': 10
        }
        
        medium_importance_keywords = {
            'khi': 12, 'ở': 12, 'đâu': 12, 'nào': 10, 'gì': 10, 'như': 10, 'thế': 10,
            'được': 8, 'không': 8, 'với': 6, 'và': 6, 'có': 8, 'là': 6, 'của': 6,
            'trong': 8, 'cho': 6, 'về': 6, 'từ': 6, 'theo': 6, 'để': 6
        }
        
        # 1. EXACT KEYWORD OVERLAP
        exact_overlap = user_words.intersection(question_words)
        norm_overlap = user_norm_words.intersection(norm_question_words)
        
        overlap_score = 0
        
        # Score exact matches với trọng số
        for word in exact_overlap:
            if word in high_importance_keywords:
                overlap_score += high_importance_keywords[word]
            elif word in medium_importance_keywords:
                overlap_score += medium_importance_keywords[word]
            else:
                overlap_score += 5  # Base score
        
        # Score normalized matches (với penalty để tránh double counting)
        for word in norm_overlap:
            if word not in exact_overlap:
                if word in high_importance_keywords:
                    overlap_score += high_importance_keywords[word] * 0.7
                elif word in medium_importance_keywords:
                    overlap_score += medium_importance_keywords[word] * 0.7
                else:
                    overlap_score += 3
        
        # 2. KEYWORD COLUMN OVERLAP
        keyword_column_score = 0
        if row['keywords']:
            keyword_words = set(str(row['keywords']).lower().split())
            keyword_overlap = user_words.intersection(keyword_words)
            
            for word in keyword_overlap:
                if word in high_importance_keywords:
                    keyword_column_score += high_importance_keywords[word] * 1.5
                elif word in medium_importance_keywords:
                    keyword_column_score += medium_importance_keywords[word] * 1.3
                else:
                    keyword_column_score += 8
        
        # 3. COVERAGE METRICS
        coverage_score = 0
        if len(user_words) > 0:
            # User coverage (how many user words are covered)
            user_coverage = len(exact_overlap) / len(user_words)
            coverage_score += user_coverage * 30
            
            # High coverage bonus
            if user_coverage > 0.8:
                coverage_score += 20
            elif user_coverage > 0.6:
                coverage_score += 10
        
        if len(question_words) > 0:
            # Question coverage (how many question words match)
            question_coverage = len(exact_overlap) / len(question_words)
            coverage_score += question_coverage * 25
        
        # 4. PARTIAL KEYWORD MATCHING
        partial_score = 0
        for user_word in user_words:
            if len(user_word) > 4:  # Only for longer words
                for q_word in question_words:
                    if len(q_word) > 4:
                        # Substring matching
                        if user_word in q_word or q_word in user_word:
                            if user_word in high_importance_keywords or q_word in high_importance_keywords:
                                partial_score += 12
                            else:
                                partial_score += 6
                            break
        
        # 5. KEYWORD DENSITY MATCHING
        density_score = 0
        user_total_words = len(user_processed.split())
        question_total_words = len(row['processed_question'].split())
        
        if user_total_words > 0 and question_total_words > 0:
            user_keyword_density = len(exact_overlap) / user_total_words
            question_keyword_density = len(exact_overlap) / question_total_words
            
            # Reward similar keyword density
            density_diff = abs(user_keyword_density - question_keyword_density)
            if density_diff < 0.2:
                density_score += 15
            elif density_diff < 0.4:
                density_score += 8
        
        # 6. CONTEXT-AWARE KEYWORD BOOSTING
        context_bonus = 0
        
        # Educational context
        edu_keywords = {'ngành', 'học', 'môn', 'đại', 'trường', 'giáo', 'dục'}
        if any(word in user_words for word in edu_keywords) and any(word in question_words for word in edu_keywords):
            context_bonus += 15
        
        # Financial context
        fin_keywords = {'phí', 'tiền', 'chi', 'giá', 'bao', 'nhiêu'}
        if any(word in user_words for word in fin_keywords) and any(word in question_words for word in fin_keywords):
            context_bonus += 15
        
        # Exam context
        exam_keywords = {'thi', 'điểm', 'phòng', 'kiểm', 'tra'}
        if any(word in user_words for word in exam_keywords) and any(word in question_words for word in exam_keywords):
            context_bonus += 15
        
        # 7. RARE KEYWORD BONUS
        rare_bonus = 0
        # Count frequency of each word in dataset to identify rare keywords
        for word in exact_overlap:
            if len(word) > 6:  # Longer words are often more specific
                rare_bonus += 8
        
        # TOTAL SCORE
        total_score = (
            overlap_score * 0.4 +           # 40% - Direct overlap
            keyword_column_score * 0.25 +   # 25% - Keyword column
            coverage_score * 0.15 +         # 15% - Coverage
            partial_score * 0.1 +           # 10% - Partial matching
            density_score * 0.05 +          # 5% - Density
            context_bonus * 0.03 +          # 3% - Context
            rare_bonus * 0.02               # 2% - Rare words
        )
        
        return total_score
    
    def hybrid_top_k_search(self, user_input: str) -> Optional[str]:
        """Enhanced Hybrid Top-K search với advanced keyword overlap"""
        candidates = self.top_k_semantic_search(user_input, k=15)  # Tăng lên 15 để có nhiều candidates hơn
        
        if not candidates:
            return None
        
        # Prepare user features
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = set(user_processed.split())
        user_words_list = user_processed.split()
        user_norm_words = set(user_normalized.split())
        
        # Re-rank candidates với enhanced scoring
        reranked_candidates = []
        
        for idx, semantic_score in candidates:
            row = self.data.iloc[idx]
            
            # === COMPONENT SCORES ===
            
            # 1. Semantic Score (normalized)
            normalized_semantic = min(semantic_score, 1.0)
            
            # 2. Advanced Keyword Overlap Score
            keyword_overlap_score = self.calculate_keyword_overlap_score(user_input, row)
            
            # 3. Fuzzy Similarity Score
            fuzzy_score = self._calculate_fuzzy_score(user_words, user_norm_words, row)
            
            # 4. Phrase Matching Score
            phrase_score = self._calculate_phrase_score(user_input, user_words_list, row)
            
            # 5. Context Score
            context_score = self._calculate_context_score(user_input, row)
            
            # === WEIGHTED COMBINATION ===
            # Tăng trọng số cho keyword overlap
            final_score = (
                normalized_semantic * 0.3 +     # 30% - Semantic
                keyword_overlap_score * 0.35 +  # 35% - Keyword Overlap (tăng từ 20%)
                fuzzy_score * 0.2 +             # 20% - Fuzzy
                phrase_score * 0.1 +            # 10% - Phrase
                context_score * 0.05            # 5% - Context
            )
            
            # === BOOSTING LOGIC ===
            # Boost cho multiple high scores
            high_scores = sum([
                normalized_semantic > 0.4,
                keyword_overlap_score > 30,
                fuzzy_score > 0.3,
                phrase_score > 20
            ])
            
            if high_scores >= 3:
                final_score *= 1.2
            elif high_scores >= 2:
                final_score *= 1.1
            
            # Boost cho perfect keyword matches
            if keyword_overlap_score > 50:
                final_score *= 1.15
            
            reranked_candidates.append((final_score, row['answer'], idx))
        
        # Return best candidate với dynamic threshold
        if reranked_candidates:
            reranked_candidates.sort(key=lambda x: x[0], reverse=True)
            best_candidate = reranked_candidates[0]
            
            # Adaptive threshold
            if len(reranked_candidates) > 1:
                top_score = best_candidate[0]
                second_score = reranked_candidates[1][0]
                
                # Lower threshold if significantly better
                if top_score > second_score * 1.4:
                    threshold = 0.12
                elif top_score > second_score * 1.2:
                    threshold = 0.18
                else:
                    threshold = 0.25
            else:
                threshold = 0.15
            
            if best_candidate[0] > threshold:
                return best_candidate[1]
        
        return None
    
    def _calculate_fuzzy_score(self, user_words: set, user_norm_words: set, row) -> float:
        """Calculate fuzzy similarity score"""
        question_words = set(row['processed_question'].split())
        norm_question_words = set(row['normalized_question'].split())
        
        if not user_words or not question_words:
            return 0.0
        
        # Multiple similarity metrics
        intersection = len(user_words & question_words)
        union = len(user_words | question_words)
        jaccard = intersection / union if union > 0 else 0
        
        containment1 = intersection / len(user_words) if len(user_words) > 0 else 0
        containment2 = intersection / len(question_words) if len(question_words) > 0 else 0
        
        norm_intersection = len(user_norm_words & norm_question_words)
        norm_union = len(user_norm_words | norm_question_words)
        norm_jaccard = norm_intersection / norm_union if norm_union > 0 else 0
        
        dice = 2 * intersection / (len(user_words) + len(question_words)) if (len(user_words) + len(question_words)) > 0 else 0
        
        return (jaccard * 0.35 + containment1 * 0.25 + containment2 * 0.15 + norm_jaccard * 0.2 + dice * 0.05)
    
    def _calculate_phrase_score(self, user_input: str, user_words_list: list, row) -> float:
        """Calculate phrase matching score"""
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        question_text = row['processed_question']
        
        score = 0
        
        # N-gram matches
        for n in range(2, min(6, len(user_words_list) + 1)):
            for i in range(len(user_words_list) - n + 1):
                ngram = ' '.join(user_words_list[i:i+n])
                if ngram in question_text:
                    score += n * 10
        
        # Substring matches
        if len(user_normalized) > 4 and user_normalized in row['normalized_question']:
            score += 35
        
        return score
    
    def _calculate_context_score(self, user_input: str, row) -> float:
        """Calculate context similarity score"""
        score = 0
        
        # Length similarity
        len_ratio = min(len(user_input), len(row['question'])) / max(len(user_input), len(row['question']))
        score += len_ratio * 0.3
        
        # Question type similarity
        user_words = set(self.preprocess_text(user_input).split())
        question_words = set(row['processed_question'].split())
        
        user_question_words = {'gì', 'nào', 'bao', 'nhiêu', 'khi', 'ở', 'đâu', 'như', 'thế', 'sao'}
        user_has_question = any(word in user_words for word in user_question_words)
        row_has_question = any(word in question_words for word in user_question_words)
        
        if user_has_question and row_has_question:
            score += 0.2
        
        return score
    
    def ensemble_semantic_search(self, user_input: str) -> Optional[str]:
        """Ensemble semantic search với top-k - keep for backward compatibility"""
        return self.hybrid_top_k_search(user_input)
    
    def phrase_match(self, user_input: str) -> Optional[str]:
        """Enhanced phrase matching với context-aware scoring"""
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = user_processed.split()
        
        # Important phrases với weights
        important_phrases = {
            'ngành học': 25, 'học phí': 25, 'phòng thi': 20, 'sinh viên': 15,
            'điểm thi': 20, 'bao nhiêu': 15, 'thời gian': 15, 'cách thức': 12,
            'như thế nào': 12, 'ở đâu': 12, 'khi nào': 12, 'đại học': 15,
            'trường fpt': 15, 'fpt university': 15, 'những gì': 10, 'làm sao': 10
        }
        
        best_match = None
        best_score = 0
        
        for idx, row in self.data.iterrows():
            question_processed = row['processed_question']
            question_normalized = row['normalized_question']
            question_words = question_processed.split()
            score = 0
            
            # === IMPORTANT PHRASE MATCHING ===
            for phrase, weight in important_phrases.items():
                if phrase in user_processed and phrase in question_processed:
                    score += weight
                elif phrase in user_normalized and phrase in question_normalized:
                    score += weight * 0.8
            
            # === MULTI-LEVEL N-GRAM MATCHING ===
            for n in range(2, min(7, len(user_words) + 1)):
                for i in range(len(user_words) - n + 1):
                    ngram = ' '.join(user_words[i:i+n])
                    
                    # Skip if ngram is too short or common
                    if len(ngram) < 4:
                        continue
                    
                    # Score based on n-gram length và rarity
                    if ngram in question_processed:
                        # Bonus for longer and rarer n-grams
                        rarity_bonus = 1.0
                        if n >= 4:
                            rarity_bonus = 1.5
                        if n >= 5:
                            rarity_bonus = 2.0
                        
                        score += n * 12 * rarity_bonus
                    
                    if ngram in question_normalized:
                        score += n * 10
            
            # === SUBSTRING MATCHING ===
            # Long substring matches
            if len(user_normalized) > 5:
                if user_normalized in question_normalized:
                    score += 35
                elif user_normalized in question_processed:
                    score += 30
            
            # Reverse substring matching
            if len(question_normalized) > 5:
                if question_normalized in user_normalized:
                    score += 25
            
            # === PATTERN MATCHING ===
            # Question word patterns
            question_patterns = [
                (['gì', 'nào'], ['gì', 'nào', 'những', 'các']),
                (['bao', 'nhiêu'], ['bao', 'nhiêu', 'mấy']),
                (['khi', 'nào'], ['khi', 'nào', 'lúc', 'thời']),
                (['ở', 'đâu'], ['ở', 'đâu', 'chỗ', 'nơi']),
                (['như', 'thế', 'nào'], ['như', 'thế', 'nào', 'cách', 'làm'])
            ]
            
            user_set = set(user_words)
            question_set = set(question_words)
            
            for user_pattern, question_pattern in question_patterns:
                if any(word in user_set for word in user_pattern):
                    if any(word in question_set for word in question_pattern):
                        score += 18
            
            # === SEMANTIC PHRASE MATCHING ===
            # Key concept matching
            key_concepts = {
                'education': ['học', 'ngành', 'môn', 'giáo', 'dục'],
                'cost': ['phí', 'tiền', 'chi', 'phí', 'giá'],
                'exam': ['thi', 'kiểm', 'tra', 'điểm'],
                'student': ['sinh', 'viên', 'học', 'sinh'],
                'time': ['thời', 'gian', 'giờ', 'phút'],
                'place': ['phòng', 'nơi', 'chỗ', 'địa']
            }
            
            user_concepts = set()
            question_concepts = set()
            
            for concept, words in key_concepts.items():
                if any(word in user_set for word in words):
                    user_concepts.add(concept)
                if any(word in question_set for word in words):
                    question_concepts.add(concept)
            
            concept_overlap = len(user_concepts.intersection(question_concepts))
            score += concept_overlap * 8
            
            # === STRUCTURE SIMILARITY ===
            # Similar question structure
            if len(user_words) > 0 and len(question_words) > 0:
                # Length similarity
                len_ratio = min(len(user_words), len(question_words)) / max(len(user_words), len(question_words))
                score += len_ratio * 10
                
                # Position-based matching (important words in similar positions)
                position_score = 0
                for i, word in enumerate(user_words):
                    if word in important_phrases or len(word) > 3:
                        # Check if word appears in similar position in question
                        relative_pos = i / len(user_words)
                        for j, q_word in enumerate(question_words):
                            if word == q_word:
                                q_relative_pos = j / len(question_words)
                                if abs(relative_pos - q_relative_pos) < 0.3:
                                    position_score += 5
                                break
                
                score += position_score
            
            # === FINAL SCORING ===
            if score > best_score and score > 15:
                best_score = score
                best_match = row['answer']
        
        return best_match
    
    def fuzzy_match(self, user_input: str) -> Optional[str]:
        """Enhanced fuzzy matching với multiple similarity metrics"""
        user_processed = self.preprocess_text(user_input)
        user_normalized = self.normalize_question(user_input)
        user_words = set(user_processed.split())
        user_norm_words = set(user_normalized.split())
        
        best_match = None
        best_similarity = 0
        
        for _, row in self.data.iterrows():
            question_words = set(row['processed_question'].split())
            norm_question_words = set(row['normalized_question'].split())
            
            if user_words and question_words:
                # Multiple similarity metrics
                
                # 1. Jaccard similarity
                intersection = len(user_words & question_words)
                union = len(user_words | question_words)
                jaccard = intersection / union if union > 0 else 0
                
                # 2. Containment similarity (both directions)
                containment1 = intersection / len(user_words) if len(user_words) > 0 else 0
                containment2 = intersection / len(question_words) if len(question_words) > 0 else 0
                
                # 3. Normalized Jaccard
                norm_intersection = len(user_norm_words & norm_question_words)
                norm_union = len(user_norm_words | norm_question_words)
                norm_jaccard = norm_intersection / norm_union if norm_union > 0 else 0
                
                # 4. Dice coefficient
                dice = 2 * intersection / (len(user_words) + len(question_words)) if (len(user_words) + len(question_words)) > 0 else 0
                
                # Combined similarity
                similarity = (
                    jaccard * 0.3 + 
                    containment1 * 0.25 + 
                    containment2 * 0.15 + 
                    norm_jaccard * 0.2 + 
                    dice * 0.1
                )
                
                # Bonus for exact matches of important words
                important_exact = 0
                important_words = ['ngành', 'học', 'phí', 'điểm', 'thi', 'phòng', 'fpt', 'môn']
                for word in important_words:
                    if word in user_words and word in question_words:
                        important_exact += 0.05
                
                similarity += important_exact
                
                if similarity > best_similarity and similarity > 0.15:
                    best_similarity = similarity
                    best_match = row['answer']
        
        return best_match
    
    def get_response(self, user_input: str) -> str:
        """Enhanced response với keyword overlap optimization"""
        if not user_input.strip():
            return "Xin chào! Tôi có thể giúp gì cho bạn?"
        
        # Primary method: Enhanced Hybrid Top-K search
        try:
            result = self.hybrid_top_k_search(user_input)
            if result:
                return result
        except Exception as e:
            print(f"Error in hybrid_top_k_search: {e}")
        
        # Secondary: Enhanced keyword overlap voting
        try:
            keyword_result = self._keyword_overlap_ensemble(user_input)
            if keyword_result:
                return keyword_result
        except Exception as e:
            print(f"Error in keyword_overlap_ensemble: {e}")
        
        # Fallback methods với enhanced weighted voting
        fallback_methods = [
            ('exact_match', 1.0),
            ('advanced_keyword_match', 0.95),  # Tăng trọng số
            ('phrase_match', 0.8),
            ('fuzzy_match', 0.7)
        ]
        
        results = {}
        
        for method_name, weight in fallback_methods:
            try:
                method = getattr(self, method_name)
                result = method(user_input)
                if result:
                    if result not in results:
                        results[result] = 0
                    results[result] += weight
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                continue
        
        # Return result với highest weighted vote
        if results:
            best_result = max(results.items(), key=lambda x: x[1])
            if best_result[1] >= 0.7:  # Confidence threshold
                return best_result[0]
        
        # Final fallback với lower threshold
        for method_name, _ in fallback_methods:
            try:
                method = getattr(self, method_name)
                result = method(user_input)
                if result:
                    return result
            except:
                continue
        
        return "Xin lỗi, tôi không tìm thấy thông tin phù hợp với câu hỏi của bạn. Bạn có thể hỏi câu hỏi khác không?"
    
    def _keyword_overlap_ensemble(self, user_input: str) -> Optional[str]:
        """Ensemble method focusing on keyword overlap"""
        # Get top candidates từ multiple methods
        candidates = []
        
        # Get semantic candidates
        semantic_candidates = self.top_k_semantic_search(user_input, k=10)
        for idx, score in semantic_candidates:
            candidates.append((idx, 'semantic', score))
        
        # Get keyword candidates by scoring all data
        user_words = set(self.preprocess_text(user_input).split())
        
        # Quick keyword screening
        for idx, row in self.data.iterrows():
            question_words = set(row['processed_question'].split())
            overlap = len(user_words.intersection(question_words))
            
            if overlap >= 2:  # At least 2 common words
                keyword_score = self.calculate_keyword_overlap_score(user_input, row)
                if keyword_score > 20:
                    candidates.append((idx, 'keyword', keyword_score))
        
        # Remove duplicates và rank all candidates
        seen_indices = set()
        unique_candidates = []
        
        for idx, method, score in candidates:
            if idx not in seen_indices:
                seen_indices.add(idx)
                # Normalize scores based on method
                if method == 'semantic':
                    normalized_score = score * 100  # Scale up semantic scores
                else:
                    normalized_score = score
                
                unique_candidates.append((idx, normalized_score))
        
        # Re-rank với keyword overlap priority
        final_candidates = []
        for idx, score in unique_candidates:
            row = self.data.iloc[idx]
            
            # Enhanced scoring
            keyword_overlap_score = self.calculate_keyword_overlap_score(user_input, row)
            fuzzy_score = self._calculate_fuzzy_score(
                set(self.preprocess_text(user_input).split()),
                set(self.normalize_question(user_input).split()),
                row
            ) * 100
            
            # Weighted final score
            final_score = (
                keyword_overlap_score * 0.6 +  # High weight on keyword overlap
                fuzzy_score * 0.25 +
                score * 0.15
            )
            
            final_candidates.append((final_score, row['answer']))
        
        # Return best candidate
        if final_candidates:
            final_candidates.sort(key=lambda x: x[0], reverse=True)
            best_score, best_answer = final_candidates[0]
            
            if best_score > 30:  # Threshold for keyword overlap ensemble
                return best_answer
        
        return None

# Flask app
app = Flask(__name__)
chatbot = None

@app.route('/', methods=['POST'])
def chat_api():
    global chatbot
    try:
        if chatbot is None:
            chatbot = AdvancedChatbot()
        
        data = request.get_json()
        user_input = data.get('chatInput', '')
        response = chatbot.get_response(user_input)
        return jsonify({'output': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def main():
    """Test function"""
    chatbot = AdvancedChatbot()
    
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
        print("Starting Advanced Chatbot API...")
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        main()
