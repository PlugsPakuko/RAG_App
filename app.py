from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import requests
import json

app = FastAPI(title="RAG Application", description="Query personal information using RAG")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="personal_info")
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "minimax-m2.5:cloud"  # Change this to your preferred model

class QueryRequest(BaseModel):
    query: str
    n_results: int = 1

class QueryResponse(BaseModel):
    query: str
    answer: str
    # relevant_documents: list[str]

def query_ollama(prompt: str) -> str:
    """Generate a response using Ollama API"""
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response generated")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Ollama API: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "RAG Application API",
        "endpoints": {
            "/query": "POST - Query the knowledge base",
            "/docs": "GET - API documentation"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system:
    1. Searches ChromaDB for relevant documents
    2. Uses Ollama to generate an answer based on the retrieved context
    """
    try:
        # Query ChromaDB for relevant documents
        results = collection.query(
            query_texts=[request.query],
            n_results=request.n_results
        )
        
        # Extract relevant documents
        if results['documents'] and len(results['documents'][0]) > 0:
            relevant_docs = results['documents'][0]
        else:
            relevant_docs = []
        
        if not relevant_docs:
            return QueryResponse(
                query=request.query,
                answer="I couldn't find any relevant information in the knowledge base.",
                relevant_documents=[]
            )
        
        context = "\n".join([f"- {doc}" for doc in relevant_docs])
        prompt = f"""Based on the following information about me, answer the question. 
If the answer cannot be found in the provided context, say so.

Context:
{context}

Question: {request.query}

Answer:"""
        
        answer = query_ollama(prompt)
        
        return QueryResponse(
            query=request.query,
            answer=answer
            # relevant_documents=relevant_docs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chromadb_collection": collection.name,
        "chromadb_count": collection.count()
    }
