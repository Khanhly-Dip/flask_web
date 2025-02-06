# Bước 1: Sử dụng image Python làm nền tảng
FROM python:3.12-slim

# Bước 2: Đặt thư mục làm việc trong container
WORKDIR /app

# Bước 3: Sao chép các file từ thư mục hiện tại vào container
COPY . /app

# Bước 4: Cài đặt các thư viện Python cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Bước 5: Mở cổng 5000 (cổng mặc định của Flask)
EXPOSE 5000

# Bước 6: Chạy ứng dụng Flask
CMD ["python", "main.py"]

services:
  mongodb:
    image: mongo:latest
    container_name: mongodb
    ports:
      - "27017:27017"
