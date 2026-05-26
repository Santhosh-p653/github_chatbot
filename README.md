# 🧠 GitHub Repository AI Chatbot (NLP-Based Code Explorer)

An interactive CLI-based AI assistant that understands and analyzes any public GitHub repository by building a semantic knowledge base of its code and documentation.

It allows you to “chat” with a repository — asking questions about functions, files, architecture, and logic — and retrieves the most relevant sections using TF-IDF + cosine similarity.

---

## 🚀 What It Does

- Fetches any **public GitHub repository**
- Recursively extracts:
  - source code files
  - documentation (README, MD files)
  - config files
- Builds a **TF-IDF vector space model**
- Enables semantic search over repository content
- Returns the most relevant file sections for user queries

---

## 🧠 How It Works (High Level)

1. User provides GitHub repo URL  
2. GitHub API is used to fetch repository structure  
3. Text/code files are extracted and cleaned  
4. TF-IDF vectorizer converts repo into numerical embeddings  
5. User query is vectorized and compared using cosine similarity  
6. Most relevant file section is returned as answer

---

## 🏗️ Tech Stack

- Python 3
- GitHub REST API
- scikit-learn (TF-IDF + Cosine Similarity)
- requests (API calls)
- NLP-based document retrieval

---

## 📦 Supported File Types

- `.py`
- `.js`, `.ts`
- `.html`, `.css`
- `.md`
- `.json`
- `.yml`, `.yaml`
- `.cpp`, `.go`, `.txt`

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

pip install -r requirements.txt