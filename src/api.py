from fastapi import FastAPI
from pydantic import BaseModel
from search_articles import (
    load_papers,
    build_vector_store,
    search_papers,
    answer_question,
    DATA_FOLDER,
)

app = FastAPI()

print("Loading papers and vector store...")
papers = load_papers(DATA_FOLDER)
collection = build_vector_store(papers)
print(f"Ready! {collection.count()} papers loaded.\n")

@app.get("/") # when someone sends a GET request to the URL /, run the funtion below
def root():
    return{
        "status":"online",
        "papers_loaded": collection.count()
        }

class QuestionRequest(BaseModel):
    question: str
    n_results: int = 3 
    
@app.post("/ask")
def ask_question(request: QuestionRequest):
    return {
        "you_asked": request.question,
        "n_results": request.n_results
    }