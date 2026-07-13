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
EMBEDDING_MODEL = "all-MiniLM-L6-v2" #converts text -> 384-dimensonal vectors.

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
    
    print(f"\n ✅ Stored {len(papers)} papers in ChromaDB!")
    return collection


def search_papers(query: str, collection: chromadb.Collection,
                  model: SentenceTransformer, n_results: int = 3 ) -> list:
    print(f"\n🔎 Searching for: '{query}'")
    
    query_embedding = model.encode(query).tolist()
    
    result = collection.query(
        query_embeddings = [query_embedding],
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
                "text":     result["documents"][0][i],
                "distance": result["distances"][0][i]
            }
        )
    print(f"    Found{len(papers_found)} relevent papers")
    return papers_found

def answer_question(query: str, relevant_papers:list)-> str:
    llm = ChatAnthropic(model="claude-haiku-4-5")
    parser = StrOutputParser()
    
    context = ""
    for i, paper in enumerate(relevant_papers,1):
        context += f"""
        paper {i}:
        Title : {paper['title']}
        Authors : {paper['authors']}
        Context : {paper['text']}
        Link : {paper['link']}
        ---
        """
    #system - sets claudes role and behaviour
    #Human - actual question + context
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """You are an expert AI research assistant with deep knowledge of machine learning and AI papers.

            Answer the user's question based ONLY on the provided research papers.
            Be specific — cite which paper says what.
            If the papers don't contain enough information, say so honestly.
            Keep your answer clear and useful for an AI engineer."""),

            ("human", """Here are the most relevant papers I found:

            {context}

            Question: {question}

            Please answer based on these papers.""")

        ]
    )

    chain = prompt | llm | parser

    print("\n Claude is generating an answer...\n")
    answer = chain.invoke({
        "context": context,
        "question": query
    })

    return answer


if __name__ == "__main__":
    print("=" * 55)
    print("  AI Research Assistant - RAG Search")
    print("=" * 55)

    papers = load_papers(DATA_FOLDER)
    if not papers:
        exit()

    collection = build_vector_store(papers)

    print("loading embedding model for search...")
    search_model = SentenceTransformer(EMBEDDING_MODEL)

    test_queries = [
        "What papers talk about robot learning and manipulation?",
        "Which papers use neural networks for prediction?",
        "What are the latest advances in machine learning?"
    ]

    for query in test_queries:
        print("\n" + "=" * 55)
        relevant_papers = search_papers(
            query       = query,
            collection  = collection,
            model       = search_model,
            n_results   = 3
        )

        print("\n Retrived papers: ")
        for i, paper in enumerate(relevant_papers, 1):
            print(f" {i}. {paper['title'][:60]}...")
            print(f" Similarity score: {1 - paper['distance']:.2%}")

        answer = answer_question(query, relevant_papers)

        print(f"Question: {query}")
        print(f"\n Answer: \n{answer}")
        print("\n" + "=" * 55)
