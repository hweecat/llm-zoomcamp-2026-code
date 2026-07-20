"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared OpenAI client.
"""

from openai import OpenAI

from gitsource import GithubRepositoryDataReader
from minsearch import Index

from rag_helper import RAGBase

import os
from dotenv import load_dotenv
from tracer import tracer

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


class RAGTraced(RAGBase):
    def rag(self, query: str) -> str:
        with tracer.start_as_current_span("rag") as span:
            return super().rag(query)
    
    def search(self, query: str) -> str:
        with tracer.start_as_current_span("search") as span:
            return super().search(query)
    
    def llm(self, query: str) -> str:
        with tracer.start_as_current_span("llm") as span:
            response = super().llm(query)
            usage = response.usage
            # for Q2
            span.set_attribute("input_tokens", usage.input_tokens)
            span.set_attribute("output_tokens", usage.output_tokens)
            return response

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id=COMMIT,
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)

client = OpenAI()
rag = RAGTraced(index=index, llm_client=client)

if __name__ == "__main__":
    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag.rag(query)
    print(answer)
