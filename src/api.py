from fastapi import FastAPI

app = FastAPI()

@app.get("/") # when someone sends a GET request to the URL /, run the funtion below
def root():
    return{"status":"online"}