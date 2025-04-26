import os
import shutil
import uuid
import sqlite3
from datetime import datetime
from typing import List, Optional
import pymupdf4llm
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import dotenv
dotenv.load_dotenv()

os.makedirs("uploads", exist_ok=True)
os.makedirs("embeddings", exist_ok=True)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize SQLite database for PDF metadata
DB_PATH = "pdf_metadata.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS pdf_metadata (
    document_id TEXT PRIMARY KEY,
    filename TEXT,
    upload_date TEXT,
    size INTEGER
)
""")
conn.commit()

app = FastAPI(title="NOTEBOOK_API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PdfUploadResponse(BaseModel):
    document_id: str
    message: str

class DocumentListResponse(BaseModel):
    document_id: str
    filename: str
    upload_date: Optional[str] = None
    size: Optional[int] = None

class QueryRequest(BaseModel):
    document_ids: List[str]  
    query: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

documents = {}  # {document_id: {"vector_store": vs, "chat_history": []}}
multi_chat_histories = {}  # {(doc_id1, doc_id2, ...): chat_history}
merged_vector_stores = {}  # {(doc_id1, doc_id2, ...): merged_vs}

def extract_text(pdf_path: str) -> str:
    try:
        return pymupdf4llm.to_markdown(pdf_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

def create_vector_store(text: str, document_id: str):
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(text)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.from_texts(chunks, embeddings)
    vs.save_local(f"embeddings/{document_id}")
    return vs

def process_pdf(file_path: str, document_id: str):
    text = extract_text(file_path)
    vs = create_vector_store(text, document_id)
    documents[document_id] = {"vector_store": vs, "chat_history": []}

def merge_vector_stores(vector_stores):
    # Merge multiple FAISS vector stores into one, avoiding duplicate doc IDs
    if not vector_stores:
        raise HTTPException(status_code=400, detail="No vector stores to merge")
    base_vs = vector_stores[0]
    for vs in vector_stores[1:]:
        # Only merge if vs is not base_vs (avoid self-merge)
        if vs is not base_vs:
            base_vs.merge_from(vs)
    return base_vs

def load_vector_store(document_id: str):
    if document_id in documents:
        return documents[document_id]["vector_store"]
    if not os.path.exists(f"embeddings/{document_id}"):
        raise HTTPException(status_code=404, detail="Document not found")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vs = FAISS.load_local(f"embeddings/{document_id}", embeddings, allow_dangerous_deserialization=True)
    documents[document_id] = {"vector_store": vs, "chat_history": []}
    return vs

@app.post("/upload/", response_model=List[PdfUploadResponse])
async def upload_pdf(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"Only PDF files are allowed: {file.filename}")
        document_id = str(uuid.uuid4())
        os.makedirs(f"uploads/{document_id}", exist_ok=True)
        file_path = f"uploads/{document_id}/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Store metadata in SQLite
        file_size = os.path.getsize(file_path)
        upload_date = datetime.utcnow().isoformat()
        cur.execute(
            "INSERT INTO pdf_metadata (document_id, filename, upload_date, size) VALUES (?, ?, ?, ?)",
            (document_id, file.filename, upload_date, file_size)
        )
        conn.commit()
        background_tasks.add_task(process_pdf, file_path, document_id)
        results.append(PdfUploadResponse(document_id=document_id, message=f"PDF '{file.filename}' uploaded and processing started"))
    return results

@app.get("/documents/", response_model=List[DocumentListResponse])
async def list_documents():
    docs = []
    # Read metadata from SQLite
    cur.execute("SELECT document_id, filename, upload_date, size FROM pdf_metadata")
    for row in cur.fetchall():
        docs.append(DocumentListResponse(
            document_id=row[0],
            filename=row[1],
            upload_date=row[2],
            size=row[3]
        ))
    return docs

@app.post("/query/", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    # Deduplicate and sort document_ids to ensure consistent cache keys
    document_ids = tuple(sorted(set(request.document_ids)))
    if len(document_ids) == 1:
        # Single document: use its vector store directly
        vs = load_vector_store(document_ids[0])
    else:
        # Multiple documents: cache merged vector store for this set
        if document_ids in merged_vector_stores:
            vs = merged_vector_stores[document_ids]
        else:
            # Always reload from disk to avoid mutating in-memory objects or deepcopy
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_stores = [
                FAISS.load_local(f"embeddings/{doc_id}", embeddings, allow_dangerous_deserialization=True)
                for doc_id in document_ids
            ]
            base_vs = vector_stores[0]
            for other_vs in vector_stores[1:]:
                base_vs.merge_from(other_vs)
            merged_vector_stores[document_ids] = base_vs
            vs = base_vs
    
    chat_key = document_ids
    chat_history = multi_chat_histories.get(chat_key, [])
    
    # Updated retrieval chain implementation
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = ChatOpenAI(temperature=0.2, model="gpt-4.1-mini")
    
    # Create system prompt for the retrieval chain
    system_prompt = """Answer the user question based on the following context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Create document chain and retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, document_chain)
    
    # Invoke the chain with the query and chat history
    result = qa_chain.invoke({"input": request.query, "chat_history": chat_history})
    
    # Extract answer and source documents
    answer = result["answer"]
    source_documents = result.get("context", [])
    
    # Update chat history
    chat_history.append((request.query, answer))
    multi_chat_histories[chat_key] = chat_history
    
    # Extract source snippets
    sources = []
    if hasattr(source_documents, "get_content"):
        sources = [source_documents.get_content()[:100] + "..."]
    elif isinstance(source_documents, list):
        sources = [doc.page_content[:100] + "..." if hasattr(doc, "page_content") else str(doc)[:100] + "..." 
                  for doc in source_documents]
    
    return {"answer": answer, "sources": sources}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}