import os
import uuid
from sentence_transformers import SentenceTransformer
import pinecone as pc
import dotenv

dotenv.load_dotenv()

# CONFIGURATION
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
ROOT_DIR = "knowledgebase"
CHUNK_SIZE = 200

# Initialize Pinecone
pinecone = pc.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pinecone.Index(PINECONE_INDEX)

# Load local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim vectors

def chunk_file(filepath, chunk_size=CHUNK_SIZE):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i+chunk_size]
        yield "".join(chunk), i+1, min(i+chunk_size, len(lines))

def embed_text(text):
    return model.encode(text).tolist()

def process_and_upload_file(filepath):
    rel_path = os.path.relpath(filepath, ROOT_DIR)
    for chunk_text, start_line, end_line in chunk_file(filepath):
        if chunk_text.strip() == "":
            continue
        try:
            embedding = embed_text(chunk_text)
            metadata = {
                "file": rel_path,
                "start_line": start_line,
                "end_line": end_line,
                "code": chunk_text
            }
            chunk_id = str(uuid.uuid4())
            index.upsert([(chunk_id, embedding, metadata)])
            print(f"Uploaded chunk {chunk_id} from {rel_path}:{start_line}-{end_line}")
        except Exception as e:
            print(f"Error processing {rel_path}:{start_line}-{end_line}: {e}")

def walk_and_process(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            process_and_upload_file(filepath)

if __name__ == "__main__":
    walk_and_process(ROOT_DIR)
    print("Done uploading all code chunks to Pinecone.")
