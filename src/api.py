from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

@app.get("/") # when someone sends a GET request to the URL /, run the funtion below
def root():
    return{"status":"online"}

class QuestionRequest(BaseModel):
    question: str
    n_results: int = 3 
    
@app.post("/ask")
def ask_question(request: QuestionRequest):
    return {
        "you_asked": request.question,
        "n_results": request.n_results
    }