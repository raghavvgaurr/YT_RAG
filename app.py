import os
import streamlit as st

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate



if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Ask the Video",
    page_icon="🎥"
)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        device="cpu"
    )


model = load_embedding_model()


class SentenceTransformerEmbeddings:

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(
            texts,
            batch_size=8,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0].tolist()


embeddings = SentenceTransformerEmbeddings(model)


# ============================================================
# VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_vectorstore():
    return Chroma(
        collection_name="youtube_rag",
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )


vectorstore = load_vectorstore()


# ============================================================
# LLM
# ============================================================

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=GROQ_API_KEY
    )


llm = load_llm()


# ============================================================
# AUGMENTATION PROMPT
# ============================================================

prompt = PromptTemplate(
    template='''You are a helpful assistant that answers questions using ONLY the provided context.

Follow these rules:

1. Use only the information present in the context.

2. Do not use outside knowledge or make up information.

3. If the context does not contain enough information to answer the question, say:
"I don't know based on the provided context."

4. Give a clear and detailed answer when the information is available.

5. Explain the answer in simple language.

6. If the context contains multiple relevant points, combine them into one coherent answer.

7. Do not mention the context, chunks, embeddings, vector database, or retrieval process.

Context:
{context}

Question:
{question}

Answer:''',

    input_variables=["context", "question"]
)


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("🎥 Ask the Video")

st.write(
    "Ask a question about the content of this YouTube video."
)


question = st.chat_input(
    "Ask something about the video..."
)


# ============================================================
# RAG PIPELINE
# ============================================================

if question:

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            # -----------------------------------------------
            # RETRIEVAL
            # -----------------------------------------------

            docs = vectorstore.similarity_search(
                query=question,
                k=5
            )

            # -----------------------------------------------
            # AUGMENTATION
            # -----------------------------------------------

            context = "\n\n".join(
                doc.page_content
                for doc in docs
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": question
            })

            # -----------------------------------------------
            # GENERATION
            # -----------------------------------------------

            response = llm.invoke(final_prompt)

        # -----------------------------------------------
        # ANSWER
        # -----------------------------------------------

        st.write(response.content)


        # ====================================================
        # REFERENCES
        # ====================================================

        st.markdown("### 📌 References")

        shown_sources = set()

        for doc in docs:

            video_url = doc.metadata.get("video_url")
            start_time = doc.metadata.get("start_time")

            if video_url is None or start_time is None:
                continue

            source_key = (
                video_url,
                start_time
            )

            if source_key in shown_sources:
                continue

            shown_sources.add(source_key)

            start_time = int(start_time)

            # Handle URLs that already contain ?
            if "?" in video_url:
                timestamp_url = (
                    f"{video_url}&t={start_time}s"
                )
            else:
                timestamp_url = (
                    f"{video_url}?t={start_time}s"
                )

            minutes = start_time // 60
            seconds = start_time % 60

            # -----------------------------------------------
            # TEXT PREVIEW
            # -----------------------------------------------

            preview = doc.page_content.strip()
            preview = preview.replace("\n", " ")

            if len(preview) > 250:
                preview = preview[:250] + "..."

            st.markdown(
                f"**🎬 {minutes}:{seconds:02d}**  "
                f"[Watch on YouTube]({timestamp_url})"
            )

            st.caption(preview)