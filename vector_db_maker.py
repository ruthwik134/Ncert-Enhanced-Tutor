import os
import pdfplumber
import re
import shutil
import time
import stat

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


# ================= CLEAN TEXT =================
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)   # remove extra spaces
    text = re.sub(r'[^a-zA-Z0-9.,()=+\-*/ ]', '', text)  # remove weird chars
    return text.strip()


# ================= HANDLE WINDOWS PERMISSION =================
def handle_remove_readonly(func, path, exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)


# ================= SAFE DELETE =================
def safe_delete(path):
    for i in range(5):
        try:
            if os.path.exists(path):
                shutil.rmtree(path, onerror=handle_remove_readonly)
                print("✅ Old vector DB deleted")
                return
        except PermissionError:
            print(f"⚠️ Attempt {i+1}: DB locked, retrying in 2 sec...")
            time.sleep(2)

    print("❌ Could not delete DB. Please close all running apps.")


# ================= EXTRACT TEXT =================
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                cleaned = clean_text(text)

                yield Document(
                    page_content=cleaned,
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "page": page_num
                    }
                )


# ================= PROCESS PDF =================
def process_pdf(pdf_path):
    documents = list(extract_text_from_pdf(pdf_path))

    # 🔥 BEST CHUNK SETTINGS
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    splits = text_splitter.split_documents(documents)

    print(f"✅ {pdf_path} → {len(splits)} chunks created")
    return splits


# ================= MULTIPLE PDFs =================
def process_multiple_pdfs(pdf_paths):
    all_docs = []
    for path in pdf_paths:
        all_docs.extend(process_pdf(path))
    return all_docs


# ================= STORE =================
def process_and_store(pdf_paths, vector_db_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    all_documents = process_multiple_pdfs(pdf_paths)

    print("⚠️ Deleting old vector DB...")
    safe_delete(vector_db_path)   # ✅ FIXED (no direct rmtree)

    print("🚀 Creating new vector DB...")

    vector_db = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=vector_db_path
    )

    print("✅ Vector DB created successfully!")


# ================= MAIN =================
def main():
    pdf_paths = [
        "NCERT-Class-12-Physics-Part-1.pdf"
    ]

    vector_db_path = "vector_db"

    process_and_store(pdf_paths, vector_db_path)


# ================= RUN =================
if __name__ == "__main__":
    main()