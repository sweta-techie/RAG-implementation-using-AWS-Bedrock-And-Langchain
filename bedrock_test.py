from langchain_community.llms import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3
import streamlit as st

# Bedrock client
bedrock_client = bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
    
)


# Use the correct text generation model
model_id = "amazon.titan-text-express-v1"

# Assign Bedrock instance to a variable with correct parameters
llm = Bedrock(
    model_id=model_id,
    client=bedrock_client,
    model_kwargs={
        'maxTokenCount': 512,   # Corrected parameter name
        'temperature': 0.5,
        'topP': 0.9             # Corrected parameter name
    }
)

# Define the chatbot function
def my_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=["language", "user_text"],
        template="You are a helpful assistant that communicates in {language}.\n\nUser: {user_text}\n\nAssistant:"
    )

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response = bedrock_chain({'language': language, 'user_text': user_text})

    return response

# Streamlit app setup
st.title("Bedrock Chatbot Demo")

# Sidebar to choose language and input question
language = st.sidebar.selectbox("Language", ["English", "Spanish", "Hindi"])

# Input text area in sidebar
user_text = st.sidebar.text_area(label="What is your question?", max_chars=500)

# Add "Send" button below the text area in sidebar
if st.sidebar.button("Send"):
    if user_text:
        try:
            response = my_chatbot(language, user_text)
            st.write(response['text'])
        except ValueError as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
