# llm-rag-tests
Travana LLM testing with RAG

### Innstall Required Libraries
----
google_cloud_aiplatform
google-generativeai
langchain
openai
python-dotenv
deeplake

### Steps to Reproduce
----
1. Install the required libraries and start running the code.
2. Prompt Engineering can be done at: `template` and `query` variables.
3. Do not change the `{context}` and `{query}` objects, it might result in Validation Errors.


### Further Steps
----
- Adding more flexibility: Instead of VertexAI, integrate ChatVertexAI.
- Exploring document similarity search methods.
- Exploring RetrievalQA chain.
