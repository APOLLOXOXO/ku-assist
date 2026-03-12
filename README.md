# KU-Assist 🎓
### AI Study Companion for Kenyatta University Students


## The Problem

Being a Telecom student at Kenyatta University means navigating three specific frustrations nobody talks about publicly:

**1. Past papers are locked behind cyber cafes and WhatsApp groups**
There is no central repository. No official archive. If you want past papers you either know the right person, pay a cyber cafe, or go without. Preparation becomes a matter of luck and connections — not effort.

**2. Lecture notes are technically accurate but humanly incomprehensible**
Units like Waveguides, Microwaves and Signal Propagation are written for someone who already understands the concepts. Dense. Jargon-heavy. Impossible to absorb at midnight before an exam. The knowledge is in the notes — but it's buried.

**3. Some lecturers practice closed-domain evaluation**
You can understand a concept perfectly. Explain it three different ways. But if your wording doesn't match exactly what's in the lecturer's notes — you fail. Not because you're wrong. Because you used your own words.

These are not edge cases. Every KU student knows all three.

KU-Assist was built to solve all three.

---

## What KU-Assist Does

KU-Assist is a **RAG (Retrieval Augmented Generation)** system that ingests your actual lecture notes and past papers — however dense, however incomplete — and answers questions based specifically on what you've uploaded.

Not generic internet answers. Your materials. Your units. Your university.

---

## Four Modes
Mode Design Philosophy

Four modes are strictly closed-domain — they only retrieve from uploaded documents. This is intentional.
KU lecturers who practice closed-domain evaluation award marks based on exact note wording. Pulling external sources into those modes would actively harm students preparing for those exams.

Deep mode is the single exception — explicitly opt-in, clearly labelled, designed for conceptual understanding not exam preparation.
The system respects the difference between learning and performing.


| Mode | What It Does | When To Use It |
|------|-------------|----------------|
| 📖 **Standard** | Answers questions directly from your uploaded notes | General revision and understanding |
| 🧠 **Simplify** | Explains dense concepts in plain human language | Waveguides at midnight. Any unit that feels impossible. |
| 💻 **Practical** | Bridges theory to working code examples | Python, programming units — when notes teach syntax but not application |
| 🎯 **Exam** | Retrieves exact wording from your lecturer's notes | For that lecturer. You know which one. |
|**Deep mode** | Understand what's being asked,retrieves from YOUR notes first,identifies what's missing or incomplete,searches the web to fill the gaps, synthesizes everything into one structured answer more as an Agentic RAG.


## How It Works

RAG works in three steps:

```
STEP 1 — INDEX
Your PDFs and notes get split into chunks
Each chunk gets converted into a vector (a number 
representation of its meaning)
Vectors get stored in ChromaDB — a local vector database

STEP 2 — RETRIEVE  
You ask a question
The system searches ChromaDB for the chunks most 
semantically similar to your question
Top matching chunks get selected

STEP 3 — GENERATE
Your question + the retrieved chunks get sent to 
a local LLM (Mistral 7B via Ollama)
The LLM reads those specific chunks and generates 
an answer grounded in your actual documents
You get an answer based on YOUR notes — not the internet
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────┐
│                   KU-ASSIST                     │
│                                                 │
│  📄 Your Documents (PDF, TXT, DOCX)             │
│              ↓                                  │
│  ✂️  Document Chunker (LangChain)               │
│     Splits docs into searchable pieces          │
│              ↓                                  │
│  🔢 Embedding Model                             │
│     (sentence-transformers/all-MiniLM-L6-v2)   │
│     Converts text chunks → vectors              │
│              ↓                                  │
│  🗄️  Vector Store (ChromaDB)                    │
│     Stores and searches embeddings locally      │
│              ↓                                  │
│  🤖 Local LLM (Mistral 7B via Ollama)           │
│     Generates answers from retrieved chunks     │
│              ↓                                  │
│  🌐 Flask API                                   │
│     Simple interface to ask questions           │
└─────────────────────────────────────────────────┘
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Core language |
| **LangChain** | RAG orchestration framework |
| **ChromaDB** | Local vector database |
| **Ollama (Mistral 7B)** | Local LLM — no API key, no cost |
| **sentence-transformers** | Text embedding model |
| **Flask** | API layer |
| **PyPDF2** | PDF ingestion |

**Everything runs locally.** No API keys. No internet dependency during inference. No cost per query.

---

## Why Local?

Most RAG tutorials use OpenAI's API. That means:
- Paying per query
- Sending your private notes to an external server
- Depending on internet connectivity
- API rate limits during exam season

KU-Assist runs entirely on your machine. Your notes stay on your machine. It works offline. It costs nothing after setup.

---

## Installation

### Prerequisites
- Python 3.10 or higher
- 8GB RAM minimum
- [Ollama](https://ollama.com) installed

### Step 1 — Clone the repo
```bash
git clone https://github.com/APOLLOXOXO/ku-assist
cd ku-assist
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Pull the LLM
```bash
ollama pull mistral
```

### Step 4 — Add your documents
```
Place your lecture notes and past papers in the /documents folder
Supported formats: PDF, TXT, DOCX
```

### Step 5 — Index your documents
```bash
python ingest.py
```

### Step 6 — Start the assistant
```bash
python app.py
```

### Step 7 — Ask a question
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain waveguide modes", "mode": "simplify"}'
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ask` | POST | Ask a question |
| `/ingest` | POST | Upload and index a new document |
| `/modes` | GET | List available modes |
| `/health` | GET | Check system status |

### Request Format
```json
{
  "question": "What are the boundary conditions for waveguides?",
  "mode": "simplify"
}
```

### Response Format
```json
{
  "answer": "Think of a waveguide like a hollow metal pipe...",
  "mode": "simplify",
  "sources": ["waveguides_lecture_3.pdf", "page 12"],
  "confidence": 0.87
}
```

---

## Challenges Faced

### 1. Chunking Strategy
The first version split documents every 500 characters — cutting sentences mid-thought and destroying context. Switched to semantic chunking with overlap: 1000 character chunks with 200 character overlap. Answer quality improved significantly.

### 2. Exam Mode Precision
Standard RAG paraphrases answers. Exam Mode needs exact text retrieval. Solved by adjusting the LLM prompt — instructing it to quote directly from retrieved chunks rather than synthesising. Temperature set to 0 for deterministic output.

### Challenge 3 — Model Memory Exceeded Available RAM

**What happened:**
Mistral 7B was the first model I tried. The ingestion pipeline worked perfectly — 29 pages chunked into 45 pieces, vectors stored in ChromaDB. But the moment I sent the first question, the server crashed with:"model requires more system memory (4.5 GiB) than is available (3.5 GiB)"
Mistral 7B needs 4.5GB of RAM just for the model itself. On a machine with 8GB total, the OS, Chrome, VS Code and ChromaDB were already consuming 4.5GB — leaving only 3.5GB free. Not enough.

**What I learned:**
RAM management is not just a hardware problem —it is an architectural decision. In production fintech and telco systems, model selection is driven partly by infrastructure constraints. 
A model that works on a 32GB cloud instance will not work on an 8GB edge device.

**The fix:**
Switched to phi3:mini — Microsoft's lightweight model that delivers strong Q&A performance at 2.3GB RAM. Same RAG pipeline, same four modes, smaller footprint.
This is the same trade-off engineers make when choosing models for mobile deployment, IoT devices and low-resource environments across Africa where high-end hardware is not always available.

**###Challenge 4** — Local LLM Response Latency
phi3:mini runs entirely on local hardware.On an 8GB RAM machine, generation takes 2-3 minutes per query. Acceptable for development and testing.
Unacceptable for a product used by hundreds of students simultaneously.
**Solution**: Migrate to Groq API for deployment.
Same open source models, cloud inference hardware,3-5 second response times, free tier available.

## Project Status

```
✅ Architecture designed
✅ README documented  
🔄 Document ingestion pipeline — in progress
🔄 ChromaDB vector store setup — in progress
⬜ Flask API endpoints
⬜ Four mode implementation
⬜ Frontend interface
⬜ Testing with real KU documents
```

---

## Real World Impact

This project addresses a gap that affects every university student in Kenya — and across Africa. The same problems exist at:

- University of Nairobi
- Strathmore University  
- JKUAT
- Moi University
- Makerere University (Uganda)
- University of Ghana

The architecture is university-agnostic. Any student can clone this, drop in their own documents and have a personalised study assistant running locally in under 10 minutes.

---

## About the Builder

**Lynne Apollo** — Fourth year Telecommunications & IT student at Kenyatta University. Building at the intersection of network virtualization and AI engineering.

- 🔗 [LinkedIn](https://linkedin.com/in/lynne-apollo-1797863b4)
- 🐙 [GitHub](https://github.com/APOLLOXOXO)
- 📅 Part of the #LockIn2026 30-day build challenge

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built out of frustration. Documented with honesty. Dedicated to every KU student who has stared at Waveguides notes at midnight and felt absolutely nothing.*
