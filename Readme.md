# 🎥 YouTube RAG — Ask the Video

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about the content of YouTube videos.

The application extracts YouTube transcripts, splits them into smaller chunks, converts the chunks into vector embeddings using a local Hugging Face embedding model, stores them in ChromaDB, and retrieves relevant information to answer user questions.

## 🚀 Current Architecture

YouTube Video
↓
YouTube Transcript
↓
Text Chunking
↓
Qwen3 Embedding Model
↓
ChromaDB
↓
Similarity Search
↓
Groq LLM
↓
Answer

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- YouTube Transcript API
- Qwen/Qwen3-Embedding-0.6B
- Sentence Transformers
- ChromaDB
- Groq
- Hugging Face

## ✨ Current Features

- Extracts transcripts from YouTube videos
- Splits transcripts into manageable chunks
- Generates local embeddings
- Stores embeddings in ChromaDB
- Performs similarity-based retrieval
- Uses an LLM to generate answers
- Provides references to the source video
- Runs through a simple Streamlit interface

## 📌 Current Status

The current version is designed around a single YouTube video.

The next development stage will expand the system to support multiple videos and larger YouTube playlists.

## 🔮 Future Improvements

- Support multiple YouTube videos
- Support complete playlists
- Improve timestamp accuracy
- Add conversation memory
- Improve follow-up question handling
- Improve retrieval accuracy
- Add better source references
- Optimize performance for larger collections

## Photos
![inspiration](<Screenshot 2026-09-18 165553.png>) ![alt text](<Screenshot 2026-09-18 165448.png>)
![Streamlit Code](<Screenshot 2026-09-22 183339.png>)
## ⚙️ Installation

youtube_RAG/
│
├── .streamlit/
│   └── config.toml
│
├── chroma_db/
│
├── app.py
├── rag.ipynb
├── requirements.txt
├── README.md
├── .env
└── .gitignore

📈 Project Roadmap
Phase 1 — Single Video
- Transcript extraction
- Chunking
- Local embeddings
- ChromaDB
- Question answering
Phase 2 — Multiple Videos
- Multiple video ingestion
- Video-level metadata
- Better source references
- Cross-video retrieval
Phase 3 — Playlist RAG
- Complete YouTube playlist ingestion
- Search across hundreds of videos
- Video and timestamp references
- Conversation-aware questions
👨‍💻 Author
Raghav Gaur


Acess Website

https://yt-ragsystem.streamlit.app/
Clone the repository:

```bash
git clone <https://github.com/raghavvgaurr/YT_RAG>
cd youtube_RAG
