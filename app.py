from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import traceback
import os
from dotenv import load_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

app = FastAPI()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found")

# ================= VECTOR DB =================
def load_vector_db(path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )

vector_db = load_vector_db("vector_db")

# 🔥 Better retriever
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# ================= LLM =================
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0   # 🔥 reduce hallucination
)

# ================= STRICT PROMPT =================
template = """
You are an NCERT Physics tutor.

STRICT RULES:
- Answer ONLY using the provided context
- Do NOT use outside knowledge
- If context is irrelevant, say: "Relevant information not found in documents"
- Do NOT guess

Context:
{context}

Question: {question}

Answer (strictly from context):
"""

prompt = ChatPromptTemplate.from_template(template)

# ================= SCHEMAS =================
class Query(BaseModel):
    question: str

class DocumentInfo(BaseModel):
    page: str
    link: str
    snippet: str

class Response(BaseModel):
    answer: str
    retrieved_documents: List[DocumentInfo]

# ================= HELPER =================
def filter_relevant_docs(query, docs):
    """Simple keyword filtering to remove garbage chunks"""
    keywords = query.lower().split()

    filtered = []
    for doc in docs:
        text = doc.page_content.lower()
        if any(word in text for word in keywords):
            filtered.append(doc)

    return filtered[:5]  # keep top relevant

# ================= API =================
@app.post("/ask", response_model=Response)
async def ask_question(query: Query):
    try:
        # 🔍 Retrieve
        docs = retriever.invoke(query.question)

        # 🔥 Filter irrelevant chunks
        docs = filter_relevant_docs(query.question, docs)

        if not docs:
            return Response(
                answer="No relevant context found in documents.",
                retrieved_documents=[]
            )

        # 📄 Clean snippets
        doc_info = []
        for doc in docs:
            snippet = doc.page_content[:250]

            doc_info.append(DocumentInfo(
                page=str(doc.metadata.get("page", "N/A")),
                link=doc.metadata.get("source", "N/A"),
                snippet=snippet
            ))

        # 🧠 Context
        context = "\n\n".join([doc.page_content for doc in docs])

        formatted_prompt = prompt.format(
            context=context,
            question=query.question
        )

        # 🤖 LLM
        response = llm.invoke(formatted_prompt)

        answer = response.content if hasattr(response, "content") else str(response)

        return Response(
            answer=answer.strip(),
            retrieved_documents=doc_info
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ================= RUN =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)