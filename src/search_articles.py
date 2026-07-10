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




 

