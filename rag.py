import io
import requests as req
import wave
import winsound
from piper import PiperVoice
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

OLLAMA = "http://localhost:11434"

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

def embed(texts):
    r = req.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts})
    r.raise_for_status()
    return r.json()["embeddings"]


class OllamaEmbed(Embeddings):
    def embed_documents(self, texts):
        return embed(texts)

    def embed_query(self, text):
        return embed([text])[0]
    

def ask_llm(prompt):
    r = req.post(f"{OLLAMA}/api/chat", json={
        "model": CHAT_MODEL, "stream": False,
        "messages": [{"role": "user", "content": prompt}]
    })
    return r.json()["message"]["content"]

def chunking():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    pages = []
    for file_path in Path("docs").glob("*.pdf"):
        reader = PdfReader(str(file_path))
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(Document(
                    page_content=text,
                    metadata={"source": str(file_path), "page": i},
                ))
    return splitter.split_documents(pages)

STORE = str(Path(__file__).parent / "store")

def vectorisation(chunked):
    db = Chroma(persist_directory=STORE, embedding_function=OllamaEmbed())
    if not db.get(limit=1)["ids"]:
        print("Base vide, ajout de", len(chunked), "chunks...")
        db.add_documents(chunked)
    return db

def retrieve(db, question, k=3):
    return db.similarity_search(question, k=k)

def rag(db, question):
    docs = retrieve(db, question)
    context= "\n\n ---- \n\n".join(d.page_content for d in docs)
    prompt = f'''Answer to this question based only to this context.
    Contexte: {context}
    Question: {question}
    '''
    return ask_llm (prompt)

VOICE = str(Path(__file__).parent / "de/de_DE-thorsten-high.onnx")


def speak(text):
    voice = PiperVoice.load(VOICE)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
    winsound.PlaySound(buf.getvalue(), winsound.SND_MEMORY)

if __name__ == "__main__":
    all_chunks = chunking()
    db = vectorisation(all_chunks)
    
    text = rag(db, "Was ist digitale Transformation? ")
    print(text)
    speak(text)

