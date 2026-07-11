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
EMBIDDING_MODEL = "all-MiniLM-L6-V2" #converts text -> 384-dimensonal vectors.

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

        


 

