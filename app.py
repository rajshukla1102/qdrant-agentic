from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
from fastembed.embedding import FlagEmbedding as TextEmbedding
from groq import Groq
import hashlib

load_dotenv()

app = FastAPI(
    title="Knowledge Base - Ingestion, Search & RAG",
    description="Complete knowledge base system with chunking and AI-powered search"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INGEST_SECRET = os.getenv("INGEST_SECRET")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "knowledge_base1"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=512)
groq_client = Groq(api_key=GROQ_API_KEY)

class IngestPayload(BaseModel):
    path: str
    repo: str
    commit: str
    deleted: bool
    content: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class RAGRequest(BaseModel):
    query: str
    limit: int = 3

class SearchResult(BaseModel):
    path: str
    title: str
    content: str
    score: float
    chunk_index: int
    total_chunks: int

def extract_frontmatter(content: str) -> tuple:
    frontmatter = {}
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1].strip()
            for line in frontmatter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip()
            content = parts[2].strip()
    return frontmatter, content

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(' '.join(chunk_words))
        start = end - overlap
        if end >= len(words):
            break
    return chunks

def generate_point_id(path: str, repo: str, chunk_index: int = 0) -> str:
    unique_string = f"{repo}:{path}:chunk_{chunk_index}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def create_collection_if_not_exists():
    """Create Qdrant collection with indexes if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        if COLLECTION_NAME not in collection_names:
            print(f"Creating collection: {COLLECTION_NAME}")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)
            )
            
            # Create payload indexes for filtering
            print("Creating payload indexes...")
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="path",
                field_schema=PayloadSchemaType.KEYWORD
            )
            qdrant_client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="repo",
                field_schema=PayloadSchemaType.KEYWORD
            )
            print(f"✅ Collection created with indexes")
        else:
            print(f"✅ Collection exists")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    print("\n" + "="*70)
    print("🚀 Starting Knowledge Base Server")
    print("="*70)
    create_collection_if_not_exists()
    print("="*70 + "\n")

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Knowledge Base - Complete System",
        "endpoints": {
            "ingestion": "/ingest - Add/update documents",
            "search": "/search - Vector similarity search",
            "rag": "/rag - AI-powered answers",
            "stats": "/stats - Collection statistics",
            "health": "/health - Health check"
        },
        "features": {
            "chunking": f"{CHUNK_SIZE} words per chunk",
            "overlap": f"{CHUNK_OVERLAP} words",
            "ai_model": "Groq - llama-3.3-70b-versatile"
        }
    }

@app.get("/health")
async def health():
    try:
        qdrant_client.get_collections()
        return {
            "status": "healthy",
            "qdrant": "connected",
            "groq": "ready" if GROQ_API_KEY else "not configured",
            "collection": COLLECTION_NAME
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Unhealthy: {str(e)}")

@app.post("/ingest")
async def ingest_document(
    payload: IngestPayload,
    x_ingest_token: Optional[str] = Header(None)
):
    if not x_ingest_token or x_ingest_token != INGEST_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    print(f"\n{'='*70}")
    print(f"📥 INGESTING: {payload.path}")
    print(f"   Repo: {payload.repo}")
    print(f"   Deleted: {payload.deleted}")
    
    if payload.deleted:
        try:
            scroll_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="path",
                        match=models.MatchValue(value=payload.path),
                    ),
                    models.FieldCondition(
                        key="repo",
                        match=models.MatchValue(value=payload.repo),
                    ),
                ]
            )
            
            scroll_result, next_page = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=scroll_filter,
                limit=100,
                with_payload=False,
                with_vectors=False,
            )
            
            point_ids = [point.id for point in scroll_result]
            
            if point_ids:
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=models.PointIdsList(points=point_ids),
                )
                print(f"🗑️  Deleted {len(point_ids)} chunks")
            else:
                print("ℹ️  No chunks found to delete")
            
            print(f"{'='*70}\n")
            return {
                "status": "success",
                "action": "deleted",
                "chunks_deleted": len(point_ids)
            }
        except Exception as e:
            print(f"❌ DELETE ERROR: {e}")
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")
    
    try:
        frontmatter, clean_content = extract_frontmatter(payload.content)
        word_count = len(clean_content.split())
        
        print(f"   Title: {frontmatter.get('title', 'N/A')}")
        print(f"   Words: {word_count}")
        
        chunks = split_into_chunks(clean_content)
        print(f"   Chunks: {len(chunks)}")
        
        points = []
        for idx, chunk in enumerate(chunks):
            embeddings = list(embedding_model.embed([chunk]))
            embedding_vector = embeddings[0].tolist()
            
            if idx == 0:
                print(f"   📊 Embedding preview: {embedding_vector[:5]}...")
            
            point_id = generate_point_id(payload.path, payload.repo, idx)
            
            metadata = {
                "path": payload.path,
                "repo": payload.repo,
                "commit": payload.commit,
                "title": frontmatter.get('title', ''),
                "description": frontmatter.get('description', ''),
                "tags": frontmatter.get('tags', ''),
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "content": chunk[:500],
                "full_chunk": chunk
            }
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding_vector,
                payload=metadata
            ))
        
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
        
        print(f"✅ SUCCESS: {len(chunks)} chunks uploaded")
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "action": "ingested",
            "path": payload.path,
            "title": frontmatter.get('title', 'N/A'),
            "word_count": word_count,
            "chunks_created": len(chunks)
        }
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"{'='*70}\n")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_documents(request: SearchRequest):
    try:
        print(f"\n{'='*70}")
        print(f"🔍 SEARCH: {request.query}")
        
        query_embedding = list(embedding_model.embed([request.query]))[0].tolist()
        
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=request.limit,
            with_payload=True,
            with_vectors=False
        ).points
        
        print(f"   Found: {len(results)} results")
        
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                path=result.payload.get("path", ""),
                title=result.payload.get("title", ""),
                content=result.payload.get("full_chunk", "")[:500],
                score=result.score,
                chunk_index=result.payload.get("chunk_index", 0),
                total_chunks=result.payload.get("total_chunks", 1)
            ))
            print(f"   ✓ {result.payload.get('title')} (score: {result.score:.3f})")
        
        print(f"{'='*70}\n")
        return search_results
        
    except Exception as e:
        print(f"❌ SEARCH ERROR: {e}\n")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/rag")
async def rag_query(request: RAGRequest):
    try:
        print(f"\n{'='*70}")
        print(f"🤖 RAG QUERY: {request.query}")
        
        query_embedding = list(embedding_model.embed([request.query]))[0].tolist()
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=request.limit,
            with_payload=True,
            with_vectors=False
        ).points
        
        print(f"   Found: {len(results)} relevant chunks")
        
        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "query": request.query
            }
        
        context_parts = []
        sources = []
        
        for idx, result in enumerate(results):
            title = result.payload.get("title", "Untitled")
            content = result.payload.get("full_chunk", "")
            
            context_parts.append(f"[Document {idx + 1}: {title}]\n{content}\n")
            sources.append({
                "title": title,
                "path": result.payload.get("path", ""),
                "score": result.score,
                "chunk": f"{result.payload.get('chunk_index', 0) + 1}/{result.payload.get('total_chunks', 1)}"
            })
            print(f"   ✓ Using: {title} (score: {result.score:.3f})")
        
        context = "\n".join(context_parts)
        
        print(f"   Calling Groq AI...")
        
        system_prompt = """You are a helpful AI assistant that answers questions based on provided documents.

Rules:
- Only use information from the documents provided
- If documents don't contain the answer, say so clearly
- Be concise and direct
- Provide specific details when available"""

        user_prompt = f"""Based on these documents:

{context}

Question: {request.query}

Provide a clear answer based only on the documents above."""

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = chat_completion.choices[0].message.content
        
        print(f"✅ Answer generated")
        print(f"{'='*70}\n")
        
        return {
            "answer": answer,
            "sources": sources,
            "query": request.query,
            "chunks_used": len(results)
        }
        
    except Exception as e:
        print(f"❌ RAG ERROR: {e}\n")
        raise HTTPException(status_code=500, detail=f"RAG failed: {str(e)}")

@app.get("/stats")
async def get_stats(x_ingest_token: Optional[str] = Header(None)):
    if not x_ingest_token or x_ingest_token != INGEST_SECRET:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    try:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        return {
            "collection": COLLECTION_NAME,
            "total_chunks": collection_info.points_count,
            "status": collection_info.status,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stats failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)