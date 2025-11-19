FROM python:3.12-slim
 
# System deps (keep minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*
 
WORKDIR /app
 
# Upgrade pip first for better wheel resolution
RUN python -m pip install --upgrade pip
 
# Install Python deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy source
COPY . .
 
# Cloud Run sets $PORT; Streamlit must bind to 0.0.0.0:$PORT
ENV PORT=8080
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=$PORT
ENV STREAMLIT_HEADLESS=true
ENV STREAMLIT_TELEMETRY=false
ENV PYTHONUNBUFFERED=1
 
# Optional: streamlit tweaks that help on serverless
#  --server.fileWatcherType none  -> reduce CPU
#  --browser.gatherUsageStats false -> turn off analytics
CMD ["bash", "-lc", "streamlit run app.py \
  --server.port ${PORT} \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.fileWatcherType none \
  --browser.gatherUsageStats false"]
 
