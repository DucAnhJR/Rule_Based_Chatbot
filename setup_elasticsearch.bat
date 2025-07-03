@echo off
echo ========================================
echo HUONG DAN CAI DAT ELASTICSEARCH
echo ========================================
echo.
echo 1. Tai Elasticsearch tu: https://www.elastic.co/downloads/elasticsearch
echo 2. Giai nen file tai ve
echo 3. Chay file bin\elasticsearch.bat (Windows) hoac bin/elasticsearch (Linux/Mac)
echo 4. Cho den khi Elasticsearch khoi dong xong (khoang 1-2 phut)
echo 5. Kiem tra bang cach mo trinh duyet va truy cap: http://localhost:9200
echo.
echo Sau khi Elasticsearch chay, ban co the chay chatbot:
echo python app.py
echo.
echo Neu khong muon dung Elasticsearch, sua trong app.py:
echo chatbot = RuleBasedChatbot(use_elasticsearch=False)
echo.
pause
