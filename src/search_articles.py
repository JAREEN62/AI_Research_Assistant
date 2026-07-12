import os
import json
import glob
from dotenv import load_dotenv

import chromadb  #the vector database that stores and searches embeddings
from sentence_transformers import SentenceTransformer #converts text into embedding vectors(numbers)
from langchain_anthropic import ChatAnthropic #Claude, which answers questions using retrived papers
from langchain_core.prompts import ChatPromptTemplate #Structures the prompts for claude
from langchain_core.output_parsers import StrOutputParser #Extracts plain text from Claude's response

load_dotenv()


DATA_FOLDER = "data"
CHROMA_FOLDER = "chroma_db" # ChromaDB saves its data here on disk so you dont re-embed everytime you run

#small fast and free, runs locally my mac.
EMBEDDING_MODEL = "all-MiniLM-L6-V2" #converts text -> 384-dimensonal vectors.

def load_papers(data_folder: str) -> list:
    all_papers=[]
    
    pattern = os.path.join(data_folder, "*.json")
    json_files = [
        f for f in glob.glob(pattern)
        if not os.path.basename(f).startswith("summaries_")
        and not os.path.basename(f).startswith("chroma")
    ]
    
    if not json_files:
        print(f"❌ No JSON files found in {data_folder}")
        print("   Run fetch_articles.py first!")
        return []
    
    for filepath in json_files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f) #reads from a file object
            
        if isinstance(data,dict) and "articles" in data:
            papers = data["articles"]
        elif isinstance(data,list):
            papers = data
        else:
            continue
        all_papers.extend(papers)
    print(f"Loaded {len(all_papers)} papers from {len(json_files)} files")
    return all_papers


def build_vector_store(papers: list)-> chromadb.Collection:
    """
    Converts each paper into an embedding vector and stores it in ChromaDB.

    WHY embeddings?
    Computers can't search text directly by meaning — but they CAN compare
    numbers. An embedding turns text into a list of numbers that captures
    its semantic meaning. Similar topics → similar numbers.

    WHY ChromaDB?
    It's a vector database — stores embeddings and lets you find the
    most similar ones to any query. Like a search engine for meaning.
    """
    print(f"\n🔢 Loading embedding model: {EMBEDDING_MODEL}")
    print("   (first run downloads ~90MB — this is normal)\n")
    
    model = SentenceTransformer(EMBEDDING_MODEL) #it converts any text into a fixed size vector of numbers.
    
    client = chromadb.PersistentClient(path=CHROMA_FOLDER) #saves chromdb to disk.
    
    collection = client.get_or_create_collection( #if collection is available from previous run use it or else create a new one
        name="research_papers",
        metadata={"hnsw:space": "cosine"} #cosine compares the angle between two vectors - perfect for comparing text meaning
    )
    existing_count= collection.count() #checking if already populated - skip re-embidding if so
    if existing_count>0:
        print(f"ChromaDB already has {existing_count} papers — skipping re-embedding")
        print("(delete chroma_db folder to re-embbed from scratch)\n")
        return collection
        
    print(f"embididng {len(papers)} papers into ChromaDB...")
    print("This may take 1-2 minutes on first run...\n")
    
    documents = [] #the text we embedd
    metadatas = [] #extra info stored alongside (not embedded)
    ids = [] #unique ID for each paper
    
    for i,paper in enumerate(papers):
        title = paper.get("title","Untitled")
        abstract = paper.get("summary", "")
        authors = paper.get("authors", [])
        link = paper.get("link","")
        
        text_to_embed = f"Title: {title}\n\nAbstract: {abstract}"
        
        documents.append(text_to_embed)
        
        metadatas.append(
            {
            "title":   title,
            "authors": ", ".join(authors[:3]),  # first 3 authors
            "link":    link, 
            }
        )
        
        ids.append(f"paper_{i}")
        
    print("Generating embiddings...")
    embeddings = model.encode(documents,show_progress_bar=True).tolist()
    
    collection.add(
        documents = documents,
        embeddings = embeddings,
        metadatas = metadatas,
        ids = ids
    )
    
    print(f"\n ✅Stored{len(papers)} [a[ers in ChromaDB!")
    return collection


def search_papers(query: str, collection: chromadb.Collection,
                  model: SentenceTransformer, n_results: int = 3 ) -> list:
    print(f"\n🔎 Searching for: '{query}'")
    
    query_embedding = model.encode(query).tolist()
    
    result = collection.query(
        query_embedding = [query_embedding],
        n_results = n_results,
        include = ["documents","metadatas","distances"]
    )
    
    papers_found = []
    for i in  range(len(result["ids"][0])):
        papers_found.append(
            {
                "title":    result["metadatas"][0][i]["title"],
                "authors":  result["metadatas"][0][i]["authors"],
                "link":     result["metadatas"][0][i]["link"],
                "text":     result["metadatas"][0][i],
                "distance": result["metadatas"][0][i]
            }
        )
    print(f"    Found{len(papers_found)} relevent papers")
    return papers_found