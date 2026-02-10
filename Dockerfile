FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py ./
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["sh", "-c", "fastmcp run server.py --transport streamable-http --host 0.0.0.0 --port ${PORT:-8000} --path /mcp"]
