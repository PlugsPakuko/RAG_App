import chromadb
from chromadb.config import Settings

# Initialize ChromaDB client with persistence
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection (will use default embedding function)
collection = chroma_client.get_or_create_collection(name="personal_info")

# Read content from aboutme.txt
with open("aboutme.txt", "r", encoding="utf-8") as f:
    content = f.read()

# Split content into chunks (splitting by newlines, filtering empty lines)
chunks = [line.strip() for line in content.split("\n") if line.strip()]

# Add documents to the collection with embeddings
# ChromaDB will automatically generate embeddings using the default embedding function
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    metadatas=[{"source": "aboutme.txt", "chunk_index": i} for i in range(len(chunks))]
)

print(f"Successfully embedded {len(chunks)} chunks from aboutme.txt into the vector database!")
print(f"Collection '{collection.name}' now contains {collection.count()} documents.")
