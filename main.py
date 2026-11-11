import shutil
from pathlib import Path

import gradio as gr

import rag
from rag import speak

DOCS = Path(__file__).parent / "docs"

# Open the vector database once at startup
db = rag.vectorisation(rag.chunking())


def upload(files):
    # Copy the selected files into docs/
    DOCS.mkdir(exist_ok=True)
    names = []
    for f in files or []:
        shutil.copy(f, DOCS / Path(f).name)
        names.append(Path(f).name)
    return "Files saved in : " + ", ".join(names) if names else "No files."

def play_audio(text):
    if text:
        speak(text)

def chat(message, history):
    # Answer the question with the RAG
    answer = rag.rag(db, message)
    return answer, answer 

with gr.Blocks(title="Virtual Lehrer") as demo:
    gr.Markdown("# Virtual Lehrer")
    with gr.Row():
        # Left: file upload
        with gr.Column(scale=1):
            files = gr.File(file_count="multiple", file_types=[".pdf"], label="PDF documents")
            result = gr.Textbox(label="Result", interactive=False)
            files.upload(upload, inputs=files, outputs=result)

        # Right: chat
        with gr.Column(scale=2):
            last_answer = gr.State("")
            bouton = gr.Button("🔊")
            bouton.click(fn=play_audio, inputs=[last_answer])
            gr.ChatInterface(
                fn=chat,
                additional_outputs=[last_answer]
                )

if __name__ == "__main__":
    demo.launch()

