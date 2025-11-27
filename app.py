from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from typing import Optional
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from fastembed.embedding import FlagEmbedding as TextEmbedding
import hashlib
import re

load_dotenv()

app = FastAPI(title="Knowledge Base Ingestion Server")

# Load environment variables
INGEST_SECRET = os.getenv("INGEST_SECRET")
print(f"Using INGEST_SECRET: {INGEST_SECRET is not None}")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "knowledge_base"

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Initialize FastEmbed (free embedding model)
embedding_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", max_length=512)

# Request model
class IngestPayload(BaseModel):
    path: str
    repo: str
    commit: str
    deleted: bool
    content: str

def extract_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown"""
    frontmatter = {}
    
    # Check if content starts with ---
    if content.startswith('---'):
        # Find the closing ---
        parts = content.split('---', 2)
        if len(parts) >= 3:
            frontmatter_text = parts[1].strip()
            
            # Simple parsing (title, description, tags, etc.)
            for line in frontmatter_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip()
            
            # Remove frontmatter from content
            content = parts[2].strip()
    
    return frontmatter, content

def generate_point_id(path: str, repo: str) -> str:
    """Generate a unique ID for each document"""
    unique_string = f"{repo}:{path}"
    return hashlib.md5(unique_string.encode()).hexdigest()

def create_collection_if_not_exists():
    """Create Qdrant collection if it doesn't exist"""
    try:
        collections = qdrant_client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if COLLECTION_NAME not in collection_names:
            print(f"Creating collection: {COLLECTION_NAME}")
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384,  # BAAI/bge-small-en-v1.5 produces 384-dim vectors
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Collection created successfully")
        else:
            print(f"✅ Collection {COLLECTION_NAME} already exists")
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize collection on startup"""
    create_collection_if_not_exists()

@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "Knowledge Base Ingestion Server",
        "collection": COLLECTION_NAME
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        qdrant_client.get_collections()
        return {"status": "healthy", "qdrant": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Qdrant connection failed: {str(e)}")

@app.post("/ingest")
async def ingest_document(
    payload: IngestPayload,
    x_ingest_token: Optional[str] = Header(None)
):
    """Main ingestion endpoint"""
    
    # Verify authentication token
    if not x_ingest_token or x_ingest_token != INGEST_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    
    print(f"\n{'='*70}")
    print(f"📥 Received: {payload.path}")
    print(f"   Repo: {payload.repo}")
    print(f"   Commit: {payload.commit}")
    print(f"   Deleted: {payload.deleted}")
    
    # Generate unique point ID
    point_id = generate_point_id(payload.path, payload.repo)
    
    # Handle deletion
    if payload.deleted:
        try:
            qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=[point_id]
            )
            print(f"🗑️  Deleted from Qdrant")
            print(f"{'='*70}\n")
            return {
                "status": "success",
                "action": "deleted",
                "path": payload.path,
                "point_id": point_id
            }
        except Exception as e:
            print(f"❌ Error deleting: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete: {str(e)}")
    
    # Process the document
    try:
        # Extract frontmatter and clean content
        frontmatter, clean_content = extract_frontmatter(payload.content)
        
        print(f"   Title: {frontmatter.get('title', 'N/A')}")
        print(f"   Content length: {len(clean_content)} chars")
        
        # Generate embedding using FastEmbed
        embeddings = list(embedding_model.embed([clean_content]))
        embedding_vector = embeddings[0].tolist()
        
        print(f"   Embedding size: {len(embedding_vector)}")
        
        # Prepare metadata
        metadata = {
            "path": payload.path,
            "repo": payload.repo,
            "commit": payload.commit,
            "title": frontmatter.get('title', ''),
            "description": frontmatter.get('description', ''),
            "tags": frontmatter.get('tags', ''),
            "content": clean_content[:1000],  # Store first 1000 chars for preview
            "full_content": payload.content  # Store full content
        }
        
        # Create point for Qdrant
        point = PointStruct(
            id=point_id,
            vector=embedding_vector,
            payload=metadata
        )
        
        # Upsert to Qdrant (insert or update)
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=[point]
        )
        
        print(f"✅ Successfully ingested to Qdrant")
        print(f"{'='*70}\n")
        
        return {
            "status": "success",
            "action": "ingested",
            "path": payload.path,
            "point_id": point_id,
            "title": frontmatter.get('title', 'N/A'),
            "embedding_size": len(embedding_vector)
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"{'='*70}\n")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/stats")
async def get_stats(x_ingest_token: Optional[str] = Header(None)):
    """Get collection statistics"""
    
    # Verify authentication
    if not x_ingest_token or x_ingest_token != INGEST_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing authentication token")
    
    try:
        collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
        return {
            "collection": COLLECTION_NAME,
            "points_count": collection_info.points_count,
            "status": collection_info.status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)