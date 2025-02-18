import os
import hashlib
import json
import threading
import uuid
from langchain_cohere import CohereEmbeddings
from langchain_chroma import Chroma
import numpy as np
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

TRACKER_DIR = "C:\\Users\\rudra\\Desktop"
SNAPSHOT_FILE = os.path.join(TRACKER_DIR, "desktop_tracker.json")
DB_PATH = "C:\\Users\\rudra\\Desktop\\File_Finder\\Common_DB"

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text:latest",
)

def load_vector_db():
    if os.path.isdir(DB_PATH):
        print("🔄 Loading existing vector database...")
        return Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    else:
        print("❌ Database directory does not exist. Returning None.")
        return None

# Function for similarity search in the vector database
def similarity_search(query, vectorstore):
    print(f"🔍 Performing similarity search for: {query}")
    search_results = vectorstore.similarity_search(query, k=5)  
    if search_results:
        print("📄 Similar Results Found:")
        for result in search_results:
            print(f"  - {result.metadata['source']}: {result.page_content[:200]}...")  # Adjust the preview of text
    else:
        print("❌ No similar results found.")
    return search_results

def remove_file_chunks(vectorstore: Chroma, file_path):
    docs = vectorstore.get(where={"source": file_path})
    doc_ids = docs["ids"]
    if doc_ids:
        vectorstore.delete(doc_ids)
        print(f"🗑️ Removed file chunks for: {file_path}")
    else:
        print(f"❌ No documents found for the given file path: {file_path}")

def add_db(vectorstore: Chroma, filepath):
    print(f"📄 Loading file: {filepath} into vector DB...")
    loader = TextLoader(filepath)
    
    try:
        docs = loader.load()
    except UnicodeDecodeError:
        print(f"❌ Error decoding file: {filepath}. Skipping this file.")
        return
    except Exception as e:
        print(f"❌ Error loading file {filepath}: {e}")
        return

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = text_splitter.split_documents(docs)

    # Check if docs is empty
    if not docs:
        print(f"❌ No valid documents found in {filepath}. Skipping this file.")
        return

    # Generate unique ids for each document
    ids = [str(uuid.uuid4()) for _ in docs]

    # Ensure the number of ids matches the number of documents
    if len(ids) != len(docs):
        print(f"❌ Mismatch between number of documents and ids for {filepath}. Skipping this file.")
        return

    # Add documents to the vector store
    vectorstore.add_documents(documents=docs, ids=ids)
    print(f"✅ File: {filepath} added to vector DB.")

def update_db(vectorstore: Chroma, filepath):
    print(f"🔄 Updating database for: {filepath}...")
    remove_file_chunks(vectorstore, filepath)
    add_db(vectorstore, filepath)
    print(f"✅ File: {filepath} has been updated successfully.")

def get_file_hash(filepath):
    """Generate a SHA-256 hash for file content."""
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            while chunk := f.read(4096):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"❌ Error hashing {filepath}: {e}")
        return None

def create_snapshot(directory):
    """Creates a snapshot of all files in the directory."""
    print("🛠️ Creating snapshot of the directory...")
    snapshot = {}
    for root, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(root, file)
            snapshot[filepath] = {
                "size": os.path.getsize(filepath),
                "hash": get_file_hash(filepath),
            }

    with open(SNAPSHOT_FILE, "w") as f:
        json.dump(snapshot, f, indent=4)

    print("✅ Snapshot created.")

def track_directory(db: Chroma, directory):
    if not os.path.exists(directory):
        print(f"❌ Error: Directory does not exist: {directory}.")
        return

    if not os.path.exists(SNAPSHOT_FILE):
        print("❗ No previous snapshot found. Creating a new one...")
        create_snapshot(directory)
        return

    with open(SNAPSHOT_FILE, "r") as f:
        old_snapshot = json.load(f)

    new_snapshot = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file == "desktop_tracker.json":
                continue
            filepath = os.path.join(root, file)
            new_snapshot[filepath] = {
                "size": os.path.getsize(filepath),
                "hash": get_file_hash(filepath),
            }

    added = list(set(new_snapshot.keys()) - set(old_snapshot.keys()))
    removed = list(set(old_snapshot.keys()) - set(new_snapshot.keys()))
    modified = {
        file
        for file in old_snapshot.keys() & new_snapshot.keys()
        if old_snapshot[file]["hash"] != new_snapshot[file]["hash"]
    }

    if added or removed or modified:
        with open(SNAPSHOT_FILE, "w") as f:
            json.dump(new_snapshot, f, indent=4)
        print("\n🔄 Snapshot updated.")

    if added:
        print("🟢 New Files Added:")
        for file in added:
            print(f"  - {file}")

    if removed:
        print("🔴 Files Removed:")
        for file in removed:
            print(f"  - {file}")

    if modified:
        print("🟡 Modified Files:")
        for file in modified:
            print(f"  - {file}")
    
    valid_ext = ["py", "txt", "java"]
    for file in removed:
        if file.split(".")[-1] in valid_ext:
            remove_file_chunks(db, file)
            print(f"🗑️ File: {file} removed from DB.")

    for file in added:
        if file.split(".")[-1] in valid_ext:
            add_db(db, file)
            print(f"📥 File: {file} added to DB.")

    for file in modified:
        if file.split(".")[-1] in valid_ext:
            update_db(db, file)
            print(f"🔄 File: {file} updated in DB.")

# Load the vector database once before the loop
db = load_vector_db()

# If the database isn't loaded properly, exit
if db is None:
    print("❌ Exiting: Could not load vector database.")
else:
    # Function to run directory monitoring in the background
    def monitor_directory():
        while True:
            track_directory(db, TRACKER_DIR)

    # Start the directory monitoring in a background thread
    directory_thread = threading.Thread(target=monitor_directory, daemon=True)
    directory_thread.start()

    # Continuously accept user input for similarity search
    while True:
        query = input("🔍 Enter search query: ")
        if query.lower() == "exit":
            print("❌ Exiting program.")
            break
        similarity_search(query, db)
