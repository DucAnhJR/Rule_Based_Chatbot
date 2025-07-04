"""
Progressive Test - Kiểm tra accuracy theo từng bước cải tiến
"""
import pandas as pd
from app import AdvancedChatbot
import time

def run_accuracy_test(sample_size=50):
    """Chạy test accuracy trên sample nhỏ để nhanh hơn"""
    print(f"=== Testing Accuracy on {sample_size} questions ===")
    
    # Load test data
    test_df = pd.read_excel('test.xlsx')
    test_sample = test_df.head(sample_size)
    
    # Initialize chatbot
    chatbot = AdvancedChatbot()
    
    # Test accuracy
    correct = 0
    total = sample_size
    
    start_time = time.time()
    
    for idx, row in test_sample.iterrows():
        question = str(row['question']).strip()
        expected = str(row['answer']).strip()
        
        try:
            chatbot_response = chatbot.get_response(question)
            is_correct = chatbot_response.lower() == expected.lower()
            
            if is_correct:
                correct += 1
                
        except Exception as e:
            print(f"Error on question {idx}: {e}")
    
    end_time = time.time()
    accuracy = (correct / total) * 100
    
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    print(f"Time: {end_time - start_time:.2f}s")
    
    return accuracy

if __name__ == "__main__":
    accuracy = run_accuracy_test()
    if accuracy >= 60:
        print("✅ TARGET REACHED: 60%+")
    else:
        print(f"❌ Need {60 - accuracy:.1f}% more")
