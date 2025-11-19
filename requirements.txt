streamlit>=1.37
pdfminer.six>=20231228
sentence-transformers>=3.0.1
scikit-learn>=1.5.1
groq
python-dotenv>=1.0.1
numpy>=1.26
scipy>=1.11
torch>=2.2 ; sys_platform != "darwin"

# PDF + visualization
reportlab>=4,<5
pillow>=10
matplotlib>=3.8
pandas>=2.1
python-dateutil>=2.9

# LangChain + Groq
langchain>=0.3
langchain-core>=0.3
langchain-groq>=0.1.0

# API server (optional)
fastapi>=0.111
uvicorn>=0.30

# Azure cloud storage
azure-storage-blob>=12.19

# Pydantic (Structured Output Schema)
pydantic>=2.7

simple-salesforce
