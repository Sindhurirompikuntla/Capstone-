# How to Use - Sales Transcript Analysis Agent

## 🚀 Quick Start

### 1. Start the API Server
```bash
python run_api.py
```

### 2. Open the Web Interface
Open in your browser:
```
examples/upload_demo.html
```

### 3. Analyze Transcripts
- **Text**: Select "Text Transcript", paste your conversation, click "Analyze"
- **Audio**: Select "Audio File", upload MP3/WAV/M4A/OGG, click "Analyze"

---

## 📁 Project Structure

```
Captsone/
├── README.md              Complete documentation
├── HOW_TO_USE.md         This file
│
├── config/               Configuration
│   ├── config.yaml      Settings
│   ├── prompts.yaml     LLM prompts
│   └── .env             Your credentials
│
├── src/                 Source code
│   ├── agent/          Analysis logic
│   ├── api/            FastAPI app
│   └── utils/          Utilities
│
├── data/               Sample data
│   └── text/          3 sample transcripts
│
├── examples/           Examples & Demos
│   ├── upload_demo.html      ⭐ Main web interface
│   ├── demo.html             Alternative interface
│   ├── analyze_text.py       Python example
│   └── api_client_example.py API examples
│
├── notebooks/          Jupyter tutorials
│
└── run_api.py         Start the server
```

---

## 🎯 What You Can Do

### Option 1: Web Interface (Easiest)
1. Start API: `python run_api.py`
2. Open: `examples/upload_demo.html`
3. Upload text or audio
4. Get instant analysis

### Option 2: Python Script
```bash
python examples/analyze_text.py
```

### Option 3: API Directly
```bash
# Start server
python run_api.py

# Use API at http://localhost:8000/docs
```

### Option 4: Jupyter Notebooks
```bash
jupyter notebook
# Open notebooks/01_basic_usage.ipynb
```

---

## 📊 What You Get

The system analyzes sales conversations and extracts:
- ✅ **Requirements** - Client needs with priorities
- ✅ **Recommendations** - Product suggestions
- ✅ **Summary** - Overview, pain points, opportunities
- ✅ **Action Items** - Next steps with owners
- ✅ **Sentiment** - Conversation tone

---

## 🔧 Configuration

Your credentials are in `config/.env`:
- Azure OpenAI credentials
- Zilliz Cloud (Milvus) credentials

---

## 🆘 Troubleshooting

### API not loading?
```bash
# Check if server is running
python run_api.py
```

### Analysis fails?
- Check Azure OpenAI credentials in `config/.env`
- Verify deployment names match your Azure OpenAI deployments

### Upload not working?
- Make sure API server is running
- Check file size (max 25MB for audio)

---

## 📚 More Information

- **README.md** - Complete documentation
- **API Docs** - http://localhost:8000/docs (when server running)

---

**Start using**: Open `examples/upload_demo.html` in your browser! 🚀

