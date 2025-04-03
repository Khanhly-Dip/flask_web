<<<<<<< HEAD
from functools import lru_cache

import pymongo
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForTokenClassification,pipeline
from nltk.stem import PorterStemmer
import re
import torch
import logging
import pymongo
import random

# Kết nối MongoDB (đảm bảo MongoDB đã được chạy và cấu hình đúng)

# đã ổi dòng này
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['mydb']
collection = db['procedures']

documents_cursor = collection.find({}, {"name": 1, "content": 1, "_id": 0})

# Chuyển đổi kết quả truy vấn thành danh sách
documents = [
    {
        "name": doc["name"],
        "content": doc["content"],
        "id": index + 1
    }
    for index, doc in enumerate(documents_cursor)
]

greeting_keywords = ["xin chào", "chào bạn", "chào", "ai có thể trợ giúp tôi", "giúp tôi", "tôi muốn biết", "có ai ở đây không?", "chào", "hello", "hi",
                      "chào anh", "chào chị", "tôi cần trợ giúp"]

greeting_responses = [
    "Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?",
    "Chào bạn, tôi là chatbot Mira, sẵn sàng giúp bạn!",
    "Xin chào! Chatbot Mira sẵn sàng trợ giúp bạn!",
    "Chào bạn! Bạn đang cần biết thêm các thông tin về thủ tục hành chính?",
    "Xin chào! Tôi là chatbot Mira trợ giúp bạn về các thông tin thủ tục hành chính"
]
default_response = "Tôi không hiểu ý bạn lắm. Nếu có bất kỳ câu hỏi nào về thủ tục hành chính, hãy cho tôi biết."

content_keys = ["Tên tài liệu", "Trạng thái tài liệu","Trình tự thực hiện", "Đối tượng thực hiện", "Cách thức thực hiện", "Thành phần hồ sơ",
                    "Căn cứ pháp lý", "Biểu mẫu đính kèm", "phí", "Lệ phí", "Yêu cầu điều kiện", "Số bộ hồ sơ", "Kết quả thực hiện",
                    "Thời hạn giải quyết", "Cơ quan thực hiện", "Cơ quan ban hành",
                    "Cơ quan phối hợp", "Thủ tục hành chính liên quan"]

with open('E:/File_Dowload/vietnamese-stopwords-master/vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
    vietnamese_stopwords = set(f.read().splitlines())
vietnamese_stopwords = [word.strip() for word in vietnamese_stopwords if word.strip()]

def preprocess_text(text):
    text = re.sub(r"[àáạảãâầấậẩẫăằắặẳẵ]", "a", text)
    text = re.sub(r"[èéẹẻẽêềếệểễ]", "e", text)
    text = re.sub(r"[ìíịỉĩ]", "i", text)
    text = re.sub(r"[òóọỏõôồốộổỗơờớợởỡ]", "o", text)
    text = re.sub(r"[ùúụủũưừứựửữ]", "u", text)
    text = re.sub(r"[ỳýỵỷỹ]", "y", text)
    text = re.sub(r"[đ]", "d", text)
    return text.strip()

processed_cache = {}

# Xử lý tình huống chào hỏi
def is_greeting(user_input):
    user_input = user_input.strip().lower()

    if len(user_input.split()) > 5:
        return False
    for keyword in greeting_keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", user_input):
            return True

    return False

def is_non_understandable(user_input):
    # Kiểm tra nếu đầu vào chỉ gồm các ký tự không có ý nghĩa (chẳng hạn dấu câu, các ký tự đặc biệt)
    user_input = user_input.strip()
    if not user_input or re.match(r'^[^\w\s]+$', user_input):  # Các ký tự không phải chữ cái hoặc số
        return True
    return False

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    # Tách từ (tokenization)
    tokens = word_tokenize(sentence)
    print("-- Tokenized:", tokens)
    return tokens

# ---------------
@lru_cache(maxsize=1000)
def preprocess_text_cached(text):
    if text in processed_cache:
        return processed_cache[text]
    processed_text = preprocess_text(text)  # Tiền xử lý văn bản (có thể thêm các bước tiền xử lý khác)
    processed_cache[text] = processed_text
    return processed_text
# ---------------

@lru_cache(maxsize=1000)
def ner_extraction_transformers(text):
    print("Văn bản đầu vào:", text)
    ner_results = nlp_ner(text)
    named_entities = []

    # Gộp các thực thể và lưu kết quả
    for entity in ner_results:
        named_entities.append({
            "entity": entity["entity_group"],
            "text": entity["word"],
            "score": entity["score"]
        })

    print("Thực thể nhận diện:", named_entities)
    return named_entities

ps = PorterStemmer()
def generate_ngrams(tokens, n):

   return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def is_meaningful_text(text):
    text = text.strip()
    meaningless_words = {"vậy", "và", "thì", "sao", "à", "ừ", "ờ", "nhé", "sao?", "muốn","còn", "về"}
    tokens = text.split()
    meaningful_tokens = [word for word in tokens if word not in meaningless_words]
    return " ".join(meaningful_tokens)

# Tăng điểm cho từ khóa yêu cầu
def prioritize_keywords(matched_keywords, important_keywords):
    prioritized = []
    seen = set()
    for keyword, sim in matched_keywords:
        if keyword not in seen:
            seen.add(keyword)
            if keyword in important_keywords:
                prioritized.append((keyword, sim * 1.5))  # Tăng điểm
            else:
                prioritized.append((keyword, sim))
    return prioritized

def find_best_keyword(tokens, content_keys, top_n=2, threshold=0.55, important_keywords=None):
    if important_keywords is None:
        important_keywords = ["Phí", "Lệ phí"]
    print("token", tokens)
    trigrams = generate_ngrams(tokens, n=3)
    fourgrams = generate_ngrams(tokens, n=4)
    all_tokens = tokens + trigrams + fourgrams

    vectorizer = TfidfVectorizer()

    if not all(isinstance(key, str) for key in content_keys):
        print("Lỗi: content_keys cần phải là danh sách các chuỗi.")
        return []
    # Tạo ma trận TF-IDF cho content_keys
    tfidf_matrix = vectorizer.fit_transform(content_keys)

    # Tính TF-IDF cho đầu vào người dùng
    user_tfidf = vectorizer.transform([" ".join(all_tokens)])

    # Tính độ tương đồng cosine giữa đầu vào người dùng và tất cả các nội dung trong content_keys
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    content_keys = list(content_keys)

    # Lọc các tài liệu có độ tương đồng lớn hơn ngưỡng
    matched_keywords = []

    for idx, sim in enumerate(cosine_similarities):
        print("content", content_keys[idx], "check ", sim)
        if sim > threshold:
            print("pass")
            matched_keywords.append((content_keys[idx], sim))
        if content_keys[idx] in important_keywords and sim >= 0.18:
            matched_keywords.append((content_keys[idx], sim))

    # Sắp xếp từ khóa theo độ tương đồng giảm dần
    matched_keywords = sorted(matched_keywords, key=lambda x: x[1], reverse=True)

    # Ưu tiên các từ khóa quan trọng
    prioritized_keywords = prioritize_keywords(matched_keywords, important_keywords)

    # Chọn top_n từ khóa sau khi ưu tiên
    top_keywords = [keyword for keyword, sim in prioritized_keywords[:top_n]]

    return top_keywords


def find_content_by_keyword(user_input, documents, best_matched):
    content_keys = {key for doc in documents if "content" in doc for key in doc["content"].keys()}
    processed_tokens = user_input.split()
    best_matches = find_best_keyword(processed_tokens, content_keys)

    result = []
    if best_matches:
        content = best_matched.get("content", {})
        seen = set()

        for best_match in best_matches:
            for second_key, value in content.items():

                if isinstance(second_key, str) and isinstance(value, str):
                    second_key_processed = preprocess_text(second_key).lower()
                    value_processed = preprocess_text(value).lower()
                    best_match_processed = best_match.lower()

                    if (best_match_processed == second_key_processed or
                            best_match_processed == value_processed or
                            best_match in second_key or best_match in value):

                        if best_match in second_key:

                            content_pair = (second_key, value)
                            if content_pair not in seen:
                                seen.add(content_pair)
                                result.append(f"{second_key}: {value}")

    if not best_matched or not result:
        content = best_matched.get("content", {})
        for key, value in content.items():
            if isinstance(key, str) and isinstance(value, str):
                result.append(f"{key}: {value}")

    if result:
        return "\n".join(result)
    else:
        return "Không tìm thấy thông tin phù hợp"


def find_best_match(user_input, documents):
    # Kiểm tra và xử lý chào hỏi
    if is_greeting(user_input):
        return random.choice(greeting_responses), None, ""

    # Kiểm tra chuỗi không thể hiểu được
    if is_non_understandable(user_input):
        return default_response, "", ""

    matched_keywords = []
    remaining_part = user_input

    # Tìm và loại bỏ từ khóa trong chuỗi
    for keyword in content_keys:
        if keyword.lower() in remaining_part.lower():
            matched_keywords.append(keyword)
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            remaining_part = pattern.sub("", remaining_part).strip()

    # Kết quả từ khóa và phần còn lại
    matched_keywords_str = " ".join(matched_keywords)
    print("Matched keywords:", matched_keywords_str)
    print("Remaining part after matching:", remaining_part)

    old_best_match = session["context"][-1]["best_match"] if session.get("context") else None
    remaining_part = is_meaningful_text(remaining_part)
    print("chuoi da chuan hoa", remaining_part)
    # Nếu không có vế sau mà có vế trước
    if not remaining_part and matched_keywords_str:
        session["context"] = [{
            "best_match": old_best_match
        }]
    # Nếu không có vế trước và có vế sau
    elif not matched_keywords_str and remaining_part:
        session["context"] = [{
            "best_match": remaining_part
        }]
    # Nếu
    elif matched_keywords_str and remaining_part != old_best_match:
        session["context"] = [{
            "best_match": remaining_part
        }]
    elif matched_keywords_str :
        session["context"] = [{
            "best_match": remaining_part
        }]
    elif not remaining_part and not matched_keywords_str:
        return "default_response", "", ""
    elif (remaining_part != old_best_match or not remaining_part) and not matched_keywords_str :
        # Đặt lại ngữ cảnh nếu kết quả mới khác
        session["context"] = [{
            "best_match": remaining_part
        }]
    else:
        # Thêm thông tin vào ngữ cảnh nếu giống
        session["context"].append({
            "best_match": remaining_part,
        })

    # print("Session cintent", session["context"])

    session.modified = True
    MAX_CONTEXT_LENGTH = 5
    if len(session["context"]) > MAX_CONTEXT_LENGTH:
        session["context"].pop(0)

    context_best_matches = [entry["best_match"] for entry in session["context"] if "best_match" in entry]

    # Kết hợp các giá trị "best_match" thành một chuỗi duy nhất (nếu cần)
    context_combined = " ".join(context_best_matches)
    print("combined", context_combined)

    # Tạo bộ vector hóa TF-IDF
    vectorizer = TfidfVectorizer()
    print("phần này chưa có lỗi 1")
    tfidf_matrix = vectorizer.fit_transform([doc["name"] for doc in documents])
    user_tfidf = vectorizer.transform([context_combined])
    print("phần này chưa có lỗi 2")
    # Tính độ tương đồng cosine
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    best_match_index = cosine_similarities.argmax()
    similarity_score = cosine_similarities[best_match_index]

    # Lưu thông tin vào session
    best_match = documents[best_match_index]["name"]
    detailed_info = documents[best_match_index].get("details", "")
    session["best_match"] = best_match
    session["similarity_score"] = similarity_score
    session["details"] = detailed_info
    session.modified = True

    # NẾU KHÔNG TÌM THẤY KẾT QUẢ PHÙ HỢP
    if similarity_score < 0.3:
        return "Tôi không hiểu ý bạn lắm. Nếu có bất kỳ câu hỏi nào về thủ tục hành chính, hãy cho tôi biết.", None, ""

    # Trả về kết quả gần đúng nhất
    best_matched = documents[best_match_index]

    result_with_entities = []
    # Nếu không có kết quả từ NER, tìm kiếm từ khóa trong content
    if not result_with_entities:
        detailed_info = find_content_by_keyword(" ".join(matched_keywords), documents, best_matched)
        print("detail, inforqưeq12", detailed_info)

    else:
        detailed_info = "\n".join(result_with_entities)

    detailed_info = detailed_info.replace("\n", "<br>")
    # print("Final result:", detailed_info)

    return best_matched["name"], similarity_score, detailed_info


# mmmmmmmmmmmmmmmmmmm


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("Form data:", request.form)
        # Lấy dữ liệu người dùng nhập vào
        user_input = request.form['user_input']

        try:
            # Gọi hàm tìm kiếm
            best_match, similarity_score, detailed_info = find_best_match(user_input, documents)
        except ValueError:
            # Xử lý lỗi trả về không đúng số lượng giá trị
            return jsonify({
                'response': "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."
            }), 500

        # Kiểm tra nếu không tìm được kết quả
        if not best_match:
            best_match = "Không tìm thấy kết quả phù hợp."
            detailed_info = "Không có thông tin chi tiết."
        return jsonify({
            'best_match': best_match,
            'detailed_info': detailed_info
        })

        # Xử lý khi GET request
    return render_template('index.html')


if __name__ == '__main__':

    logging.getLogger("transformers").setLevel(logging.ERROR)
    # kiểm tra CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải DistilPhoBERT
    tokenizer = BertTokenizer.from_pretrained("vinai/phobert-base")
    model = BertForTokenClassification.from_pretrained("vinai/phobert-base").to(device)
    nlp_ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    app.run(host="127.0.0.1", port=5000, debug=True)

# nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
=======
from functools import lru_cache
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForTokenClassification,pipeline
from nltk.stem import PorterStemmer
import re
import torch
import logging
import pymongo
import random

# đã thay đổi rồi nhé


# Kết nối MongoDB (đảm bảo MongoDB đã được chạy và cấu hình đúng)
#đã thay đổi
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['mydb']
collection = db['procedures']


#Thay đổi rồi mà sao không chụ đổi
documents_cursor = collection.find({}, {"name": 1, "content": 1, "_id": 0})

# Chuyển đổi kết quả truy vấn thành danh sách
documents = [
    {
        "name": doc["name"],
        "content": doc["content"],
        "id": index + 1
    }
    for index, doc in enumerate(documents_cursor)
]

greeting_keywords = ["xin chào", "chào bạn", "chào", "ai có thể trợ giúp tôi", "giúp tôi", "tôi muốn biết", "có ai ở đây không?", "chào", "hello", "hi",
                      "chào anh", "chào chị", "tôi cần trợ giúp"]

greeting_responses = [
    "Chào bạn! Tôi có thể giúp gì cho bạn hôm nay?",
    "Chào bạn, tôi là chatbot Mira, sẵn sàng giúp bạn!",
    "Xin chào! Chatbot Mira sẵn sàng trợ giúp bạn!",
    "Chào bạn! Bạn đang cần biết thêm các thông tin về thủ tục hành chính?",
    "Xin chào! Tôi là chatbot Mira trợ giúp bạn về các thông tin thủ tục hành chính"
]
default_response = "Tôi không hiểu ý bạn lắm. Nếu có bất kỳ câu hỏi nào về thủ tục hành chính, hãy cho tôi biết."

content_keys = ["Tên tài liệu", "Trạng thái tài liệu","Trình tự thực hiện", "Đối tượng thực hiện", "Cách thức thực hiện", "Thành phần hồ sơ",
                    "Căn cứ pháp lý", "Biểu mẫu đính kèm", "phí", "Lệ phí", "Yêu cầu điều kiện", "Số bộ hồ sơ", "Kết quả thực hiện",
                    "Thời hạn giải quyết", "Cơ quan thực hiện", "Cơ quan ban hành",
                    "Cơ quan phối hợp", "Thủ tục hành chính liên quan"]

with open('/mnt/d/flask_web/vietnamese-stopwords-master/vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
    vietnamese_stopwords = set(f.read().splitlines())
vietnamese_stopwords = [word.strip() for word in vietnamese_stopwords if word.strip()]

def preprocess_text(text):
    text = re.sub(r"[àáạảãâầấậẩẫăằắặẳẵ]", "a", text)
    text = re.sub(r"[èéẹẻẽêềếệểễ]", "e", text)
    text = re.sub(r"[ìíịỉĩ]", "i", text)
    text = re.sub(r"[òóọỏõôồốộổỗơờớợởỡ]", "o", text)
    text = re.sub(r"[ùúụủũưừứựửữ]", "u", text)
    text = re.sub(r"[ỳýỵỷỹ]", "y", text)
    text = re.sub(r"[đ]", "d", text)
    return text.strip()

processed_cache = {}

# Xử lý tình huống chào hỏi
def is_greeting(user_input):
    user_input = user_input.strip().lower()

    if len(user_input.split()) > 5:
        return False
    for keyword in greeting_keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", user_input):
            return True

    return False

def is_non_understandable(user_input):
    # Kiểm tra nếu đầu vào chỉ gồm các ký tự không có ý nghĩa (chẳng hạn dấu câu, các ký tự đặc biệt)
    user_input = user_input.strip()
    if not user_input or re.match(r'^[^\w\s]+$', user_input):  # Các ký tự không phải chữ cái hoặc số
        return True
    return False

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    # Tách từ (tokenization)
    tokens = word_tokenize(sentence)
    print("-- Tokenized:", tokens)
    return tokens

# ---------------
@lru_cache(maxsize=1000)
def preprocess_text_cached(text):
    if text in processed_cache:
        return processed_cache[text]
    processed_text = preprocess_text(text)  # Tiền xử lý văn bản (có thể thêm các bước tiền xử lý khác)
    processed_cache[text] = processed_text
    return processed_text
# ---------------

@lru_cache(maxsize=1000)
def ner_extraction_transformers(text):
    print("Văn bản đầu vào:", text)
    ner_results = nlp_ner(text)
    named_entities = []

    # Gộp các thực thể và lưu kết quả
    for entity in ner_results:
        named_entities.append({
            "entity": entity["entity_group"],
            "text": entity["word"],
            "score": entity["score"]
        })

    print("Thực thể nhận diện:", named_entities)
    return named_entities

ps = PorterStemmer()
def generate_ngrams(tokens, n):

   return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def is_meaningful_text(text):
    text = text.strip()
    meaningless_words = {"vậy", "và", "thì", "sao", "à", "ừ", "ờ", "nhé", "sao?", "muốn","còn", "về"}
    tokens = text.split()
    meaningful_tokens = [word for word in tokens if word not in meaningless_words]
    return " ".join(meaningful_tokens)

# Tăng điểm cho từ khóa yêu cầu
def prioritize_keywords(matched_keywords, important_keywords):
    prioritized = []
    seen = set()
    for keyword, sim in matched_keywords:
        if keyword not in seen:
            seen.add(keyword)
            if keyword in important_keywords:
                prioritized.append((keyword, sim * 1.5))  # Tăng điểm
            else:
                prioritized.append((keyword, sim))
    return prioritized

def find_best_keyword(tokens, content_keys, top_n=2, threshold=0.55, important_keywords=None):
    if important_keywords is None:
        important_keywords = ["Phí", "Lệ phí"]
    print("token", tokens)
    trigrams = generate_ngrams(tokens, n=3)
    fourgrams = generate_ngrams(tokens, n=4)
    all_tokens = tokens + trigrams + fourgrams

    vectorizer = TfidfVectorizer()

    if not all(isinstance(key, str) for key in content_keys):
        print("Lỗi: content_keys cần phải là danh sách các chuỗi.")
        return []
    # Tạo ma trận TF-IDF cho content_keys
    tfidf_matrix = vectorizer.fit_transform(content_keys)

    # Tính TF-IDF cho đầu vào người dùng
    user_tfidf = vectorizer.transform([" ".join(all_tokens)])

    # Tính độ tương đồng cosine giữa đầu vào người dùng và tất cả các nội dung trong content_keys
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    content_keys = list(content_keys)

    # Lọc các tài liệu có độ tương đồng lớn hơn ngưỡng
    matched_keywords = []

    for idx, sim in enumerate(cosine_similarities):
        print("content", content_keys[idx], "check ", sim)
        if sim > threshold:
            print("pass")
            matched_keywords.append((content_keys[idx], sim))
        if content_keys[idx] in important_keywords and sim >= 0.18:
            matched_keywords.append((content_keys[idx], sim))

    # Sắp xếp từ khóa theo độ tương đồng giảm dần
    matched_keywords = sorted(matched_keywords, key=lambda x: x[1], reverse=True)

    # Ưu tiên các từ khóa quan trọng
    prioritized_keywords = prioritize_keywords(matched_keywords, important_keywords)

    # Chọn top_n từ khóa sau khi ưu tiên
    top_keywords = [keyword for keyword, sim in prioritized_keywords[:top_n]]

    return top_keywords


def find_content_by_keyword(user_input, documents, best_matched):
    content_keys = {key for doc in documents if "content" in doc for key in doc["content"].keys()}
    processed_tokens = user_input.split()
    best_matches = find_best_keyword(processed_tokens, content_keys)

    result = []
    if best_matches:
        content = best_matched.get("content", {})
        seen = set()

        for best_match in best_matches:
            for second_key, value in content.items():

                if isinstance(second_key, str) and isinstance(value, str):
                    second_key_processed = preprocess_text(second_key).lower()
                    value_processed = preprocess_text(value).lower()
                    best_match_processed = best_match.lower()

                    if (best_match_processed == second_key_processed or
                            best_match_processed == value_processed or
                            best_match in second_key or best_match in value):

                        if best_match in second_key:

                            content_pair = (second_key, value)
                            if content_pair not in seen:
                                seen.add(content_pair)
                                result.append(f"{second_key}: {value}")

    if not best_matched or not result:
        content = best_matched.get("content", {})
        for key, value in content.items():
            if isinstance(key, str) and isinstance(value, str):
                result.append(f"{key}: {value}")

    if result:
        return "\n".join(result)
    else:
        return "Không tìm thấy thông tin phù hợp"


def find_best_match(user_input, documents):
    # Kiểm tra và xử lý chào hỏi
    if is_greeting(user_input):
        return random.choice(greeting_responses), None, ""

    # Kiểm tra chuỗi không thể hiểu được
    if is_non_understandable(user_input):
        return default_response, "", ""

    matched_keywords = []
    remaining_part = user_input

    # Tìm và loại bỏ từ khóa trong chuỗi
    for keyword in content_keys:
        if keyword.lower() in remaining_part.lower():
            matched_keywords.append(keyword)
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            remaining_part = pattern.sub("", remaining_part).strip()

    # Kết quả từ khóa và phần còn lại
    matched_keywords_str = " ".join(matched_keywords)
    print("Matched keywords:", matched_keywords_str)
    print("Remaining part after matching:", remaining_part)

    old_best_match = session["context"][-1]["best_match"] if session.get("context") else None
    remaining_part = is_meaningful_text(remaining_part)
    print("chuoi da chuan hoa", remaining_part)
    # Nếu không có vế sau mà có vế trước
    if not remaining_part and matched_keywords_str:
        session["context"] = [{
            "best_match": old_best_match
        }]
    # Nếu không có vế trước và có vế sau
    elif not matched_keywords_str and remaining_part:
        session["context"] = [{
            "best_match": remaining_part
        }]
    # Nếu
    elif matched_keywords_str and remaining_part != old_best_match:
        session["context"] = [{
            "best_match": remaining_part
        }]
    elif matched_keywords_str :
        session["context"] = [{
            "best_match": remaining_part
        }]
    elif not remaining_part and not matched_keywords_str:
        return "default_response", "", ""
    elif (remaining_part != old_best_match or not remaining_part) and not matched_keywords_str :
        # Đặt lại ngữ cảnh nếu kết quả mới khác
        session["context"] = [{
            "best_match": remaining_part
        }]
    else:
        # Thêm thông tin vào ngữ cảnh nếu giống
        session["context"].append({
            "best_match": remaining_part,
        })

    # print("Session cintent", session["context"])

    session.modified = True
    MAX_CONTEXT_LENGTH = 5
    if len(session["context"]) > MAX_CONTEXT_LENGTH:
        session["context"].pop(0)

    context_best_matches = [entry["best_match"] for entry in session["context"] if "best_match" in entry]

    # Kết hợp các giá trị "best_match" thành một chuỗi duy nhất (nếu cần)
    context_combined = " ".join(context_best_matches)
    print("combined", context_combined)

    # Tạo bộ vector hóa TF-IDF
    vectorizer = TfidfVectorizer()
    print("phần này chưa có lỗi 1")
    tfidf_matrix = vectorizer.fit_transform([doc["name"] for doc in documents])
    user_tfidf = vectorizer.transform([context_combined])
    print("phần này chưa có lỗi 2")
    # Tính độ tương đồng cosine
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    best_match_index = cosine_similarities.argmax()
    similarity_score = cosine_similarities[best_match_index]

    # Lưu thông tin vào session
    best_match = documents[best_match_index]["name"]
    detailed_info = documents[best_match_index].get("details", "")
    session["best_match"] = best_match
    session["similarity_score"] = similarity_score
    session["details"] = detailed_info
    session.modified = True

    # NẾU KHÔNG TÌM THẤY KẾT QUẢ PHÙ HỢP
    if similarity_score < 0.3:
        return "Tôi không hiểu ý bạn lắm. Nếu có bất kỳ câu hỏi nào về thủ tục hành chính, hãy cho tôi biết.", None, ""

    # Trả về kết quả gần đúng nhất
    best_matched = documents[best_match_index]

    result_with_entities = []
    # Nếu không có kết quả từ NER, tìm kiếm từ khóa trong content
    if not result_with_entities:
        detailed_info = find_content_by_keyword(" ".join(matched_keywords), documents, best_matched)
        print("detail, inforqưeq12", detailed_info)

    else:
        detailed_info = "\n".join(result_with_entities)

    detailed_info = detailed_info.replace("\n", "<br>")
    # print("Final result:", detailed_info)

    return best_matched["name"], similarity_score, detailed_info


# mmmmmmmmmmmmmmmmmmm


app = Flask(__name__)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print("Form data:", request.form)
        # Lấy dữ liệu người dùng nhập vào
        user_input = request.form['user_input']

        try:
            # Gọi hàm tìm kiếm
            best_match, similarity_score, detailed_info = find_best_match(user_input, documents)
        except ValueError:
            # Xử lý lỗi trả về không đúng số lượng giá trị
            return jsonify({
                'response': "Đã xảy ra lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại."
            }), 500

        # Kiểm tra nếu không tìm được kết quả
        if not best_match:
            best_match = "Không tìm thấy kết quả phù hợp."
            detailed_info = "Không có thông tin chi tiết."
        return jsonify({
            'best_match': best_match,
            'detailed_info': detailed_info
        })

        # Xử lý khi GET request
    return render_template('index.html')


if __name__ == '__main__':

    logging.getLogger("transformers").setLevel(logging.ERROR)
    # kiểm tra CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tải DistilPhoBERT
    tokenizer = BertTokenizer.from_pretrained("vinai/phobert-base")
    model = BertForTokenClassification.from_pretrained("vinai/phobert-base").to(device)
    nlp_ner = pipeline('ner', model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    app.run(host="127.0.0.1", port=5000, debug=True)

# nnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn
>>>>>>> dad925c ({commit_message})
