import os
import json 
import glob
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatAnthropic(model="claude-haiku-4-5")

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert AI research assistant. 
     Your job is to read academic paper abstracts and produce clear, concise summaries that a developer can quickly understand.
 
Always respond in this exact JSON format:
{{
  "one_liner": "One sentence summary of what this paper is about",
  "key_contribution": "The main thing this paper introduces or proves",
  "why_it_matters": "Why an AI engineer should care about this",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}"""),
    
    ("human", """Please summarize this resesarch paper:
     Title: {title}
     Abstract: {abstract}""")
])

chain = prompt | llm | StrOutputParser()

def load_all_articles(data_folder: str) -> list:
    """
    Loads all JSON files from your data/ folder.
    Returns a flat list of all articles across all files.
    """
    all_articles = []
    
    # find all .JSON files in the data folder
    pattern = os.path.join(data_folder,"*.json") #builds a file path for any operating system carefully.
    json_files = glob.glob(pattern)
    
    if not json_files:
         print(f"❌ No JSON files found in: {data_folder}")
         print("Run fetch_articles.py first!")
         return[]
    print(f" Found {len(json_files)} JSON file(s) to process:")
    
    for filepath in json_files:
        filename = os.path.basename(filepath)
        
        with open(filepath,"r", encoding="utf-8") as f:
            data = json.load(f)
         
        # to unwrap the articles in fetch_articles.py file    
        if isinstance(data, dict) and "articles" in data:
            articles = data["articles"]
            topic = data.get("topic", "unknown")
        elif isinstance(data, list): 
            articles = data # haldles the older format too
            
            topic = "unknown"
        else:
            articles = []
            
        print(f"   ✅ {filename} → {len(articles)} articles (topic: {topic})")
        all_articles.extend(articles) 

    return all_articles