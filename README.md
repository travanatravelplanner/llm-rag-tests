# llm-rag-tests
Travana LLM testing with RAG

### Innstall Required Libraries
----
```
google_cloud_aiplatform

google-generativeai

langchain

openai

python-dotenv

deeplake

streamlit
```
Run: ```pip install langchain openai python-dotenv deeplake google-generativeai google_cloud_aiplatform streamlit```

### Steps to Reproduce
----
1. Install the required libraries and start running the code. You should create a `.env` file to store the API keys.
2. Prompt Engineering can be done at: `_template`, `template` and `default_question` variables.
3. Do not change the `{context}`, `{chat_history}` and `{question}` objects, it might result in Validation Errors.
4. Run `streamlit run app.py`

### Further Steps
----
☑️ Adding more flexibility: Instead of VertexAI, integrate ChatVertexAI.

☑️ Exploring ConversationalRetrievalQA chain along with ConversationalBufferMemory.

- Exploring document similarity search methods.
