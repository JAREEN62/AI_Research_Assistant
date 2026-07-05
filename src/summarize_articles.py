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

