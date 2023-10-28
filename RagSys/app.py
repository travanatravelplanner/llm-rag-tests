import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatVertexAI
from langchain.embeddings import VertexAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import DeepLake
import time

# Load environment variables
load_dotenv()

# Initialize global variables
model = None
db = None
db_loaded = None

# Default values
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
You can assume the question about the most recent state of the union address.

Chat History:
{chat_history}

Follow Up Input: {question}

Standalone question:"""

template = """
            You are an expert intelligent and creative AI itinerary planner with extensive knowledge of places worldwide. Your goal is to plan an optimized itinerary for the user based on their specific interests and preferences, geographical proximity, and efficient routes to minimize travel time. To achieve this, follow these instructions:

            1. Suggest three activities along with two restaurant per day. Each activity should include the name of the place, a brief description, estimated cost, and time to visit the place.

            2. Generate a well-structured itinerary including day-to-day activities, timings to visit each location, and estimated costs for the user's reference.

            3. Take into account factors such as geographical proximity between destinations, transportation options, and other logistical considerations when planning the route.
            
            By following these guidelines, you will create a comprehensive and optimized itinerary that meets the user's expectations while ensuring minimal travel time.
          
            Use this context to generate the itinerary
            
            ==========
            {context}
            ==========

            Question: {question}

            Structure the itinerary as follows:
            {{"Name":"name of the trip", "description":"description of the entire trip", "budget":"budget of the entire thing", "data": [{{"day":1, "day_description":"Description based on the entire day's places. in a couple of words, for example: `Urban Exploration`, `Historical Exploration`, `Spiritual Tour`, `Adventurous Journey`, `Dayout in a beach`, `Wildlife Safari`, `Artistic Getaway`, `Romantic Getaway`, `Desert Safari`, `Island Hopping Adventure`...",  "places":[{{"name":"Place Name", "description":"Place Description in two lines","time_to_visit": "time to visit this place, for example: 9:00 to 10:00", "budget":"cost"}}, {{"name":"Place Name 2", "description":"Place Description 2 in two lines","time_to_visit": "time to visit this place, for example 10:30 - 13:00", "budget":"cost"}}]}}, {{"day":2, "day_description": "Description based on the entire day's places", "places":[{{"name":"Place Name","description":"Place Description in two lines","time_to_visit": "time to visit this place","budget":"cost"}}, {{"name":"Place Name 2", "description":"Place Description 2 in two lines","time_to_visit": "time to visit this place","budget":"cost"}}]}}]}}
            Note: Do not include any extra information outside this structure.

            Answer in JSON format:"""

default_question = "Generate 3 day itinerary for a trip to Hyderabad, India. Consider budget, timings and requirements. Include estimated cost and timings to visit for each activity and restaurant"

def initialize_model(db_name, embeddings):
    global model, db, db_loaded
    db_loaded = DeepLake(
        dataset_path=f"hub://travana_db/{db_name}", 
        embedding=embeddings, 
        token=os.getenv("ACTIVELOOP_TOKEN"),
        read_only=True
    )
    retriever = db_loaded.as_retriever()

    llm = ChatVertexAI(model_name='chat-bison', temperature=0.7, max_output_tokens=2004)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_docs=True
    )

    model = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        condense_question_prompt=PromptTemplate.from_template(_template),
        combine_docs_chain_kwargs={"prompt": PromptTemplate(template=template, input_variables=["context", "question"])}
    )
def main():
    global _template, template  # Add this line to access the global variables

    st.title("Travana Chatbot")

    # Sidebar for template inputs
    st.sidebar.header("Prompt Template Configuration")
    _template = st.sidebar.text_area("Conversation/Chat History Prompt:", _template)
    template = st.sidebar.text_area("System Prompt:", template)
    db_name = st.sidebar.text_input("Database Name:", "hyd_embedding")
    embeddings = VertexAIEmbeddings()

    # Upload to vector store
    uploaded_text = st.sidebar.text_area("Upload Text to Vector Store")
    if st.sidebar.button("Upload to Vector Store"):
        loader = TextLoader(uploaded_text)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        docs = text_splitter.split_documents(documents)
        dataset_path=f"hub://travana_db/{db_name}"
        db = DeepLake(embedding=embeddings, dataset_path=dataset_path, overwrite=True)
        db.add_documents(docs)

    # Initialize the model with the given db_name
    initialize_model(db_name, embeddings)

    # Chatbot UI similar to the one you provided
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("You:"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        result = model({"question": prompt})
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            assistant_response = result['answer']
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
