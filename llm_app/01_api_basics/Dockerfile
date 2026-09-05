# 基础镜像
FROM python:3.12-slim

# 创建工作目录
WORKDIR /code

# 拷贝依赖清单
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY app.py .

# 申明服务监听 8000 端口
EXPOSE 8000

# 启动执行命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

