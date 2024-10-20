# app.py

import os
import re
import time
import json
import logging
from tempfile import NamedTemporaryFile
from typing import List

import streamlit as st
import fitz  # PyMuPDF
import boto3
from botocore.exceptions import ClientError, BotoCoreError

# -----------------------------
# Configure Logging
# -----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------
# AWS Configuration
# -----------------------------
s3_client = boto3.client('s3')
textract_client = boto3.client('textract')
bedrock_client = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')
dynamodb_table = dynamodb.Table('QnA_Sessions')  # Ensure this table exists

# -----------------------------
# Streamlit Configuration
# -----------------------------
st.set_page_config(page_title="AWS RAG PDF Q&A with Amazon Titan", layout="wide")
st.title("Retrieval-Augmented Generation (RAG) PDF Q&A Application with Amazon Titan")

# -----------------------------
# Utility Functions
# -----------------------------

def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def upload_pdf_to_s3(file, bucket_name: str, object_name: str) -> bool:
    """Upload a PDF file to S3."""
    try:
        s3_client.upload_fileobj(file, bucket_name, object_name)
        logger.info(f"Uploaded {object_name} to bucket {bucket_name}.")
        return True
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Failed to upload {object_name} to bucket {bucket_name}: {e}")
        return False

def initiate_textract_job(bucket_name: str, document_name: str) -> str:
    """Start a Textract job to extract text from a PDF."""
    try:
        response = textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': document_name}}
        )
        job_id = response['JobId']
        logger.info(f"Started Textract job with ID: {job_id}")
        return job_id
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Failed to start Textract job for {document_name}: {e}")
        return ""

def check_textract_job_status(job_id: str) -> str:
    """Check the status of a Textract job."""
    try:
        response = textract_client.get_document_text_detection(JobId=job_id)
        status = response['JobStatus']
        logger.info(f"Textract job {job_id} status: {status}")
        return status
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Failed to get status for Textract job {job_id}: {e}")
        return "FAILED"

def get_textract_results(job_id: str) -> str:
    """Retrieve the extracted text from Textract results."""
    extracted_text = ""
    try:
        response = textract_client.get_document_text_detection(JobId=job_id)
        blocks = response.get('Blocks', [])
        for block in blocks:
            if block['BlockType'] == 'LINE':
                extracted_text += block['Text'] + " "
        logger.info(f"Extracted text length: {len(extracted_text)} characters.")
        return clean_text(extracted_text)
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Failed to get Textract results for job {job_id}: {e}")
        return ""

def generate_text_with_bedrock(prompt: str, model_id: str = 'amazon.titan-text-generation') -> str:
    """Generate text using Amazon Bedrock (Titan Text Generation)."""
    try:
        response = bedrock_client.invoke_model(
            ModelId=model_id,  # Replace with actual Model ID
            ContentType='application/json',
            Body=json.dumps({
                'prompt': prompt,
                'max_tokens': 500,
                'temperature': 0.7
            })
        )
        response_body = response['Body'].read().decode('utf-8')
        response_json = json.loads(response_body)
        generated_text = response_json.get('generated_text', '').strip()
        logger.info("Generated text with Bedrock.")
        return generated_text
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Failed to generate text with Bedrock: {e}")
        return ""

def store_session(session_id: str, data: dict):
    """Store session data in DynamoDB."""
    try:
        dynamodb_table.put_item(
            Item={
                'SessionID': session_id,
                **data
            }
        )
        logger.info(f"Stored data for session {session_id} in DynamoDB.")
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Failed to store session {session_id}: {e}")

def retrieve_session(session_id: str) -> dict:
    """Retrieve session data from DynamoDB."""
    try:
        response = dynamodb_table.get_item(
            Key={'SessionID': session_id}
        )
        item = response.get('Item', {})
        logger.info(f"Retrieved data for session {session_id} from DynamoDB.")
        return item
    except (ClientError, BotoCoreError) as e:
        logger.error(f"Failed to retrieve session {session_id}: {e}")
        return {}

def reset_session():
    """Reset the Streamlit session state."""
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_rerun()

# -----------------------------
# Streamlit Application Logic
# -----------------------------

def main():
    # Initialize session state variables
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(int(time.time()))
    if 'questions' not in st.session_state:
        st.session_state.questions = []
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'pdf_text' not in st.session_state:
        st.session_state.pdf_text = ""
    if 'pdf_summary' not in st.session_state:
        st.session_state.pdf_summary = ""
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'previous_num_questions' not in st.session_state:
        st.session_state.previous_num_questions = 5  # Default slider value

    # Sidebar Inputs
    st.sidebar.header("Input Options")

    # Upload PDF
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        # Reset session if a new file is uploaded
        if uploaded_file.name not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(uploaded_file.name)
            st.session_state.questions = []
            st.session_state.answers = {}
            st.session_state.pdf_text = ""
            st.session_state.pdf_summary = ""
            st.session_state.topics = []

            # Upload to S3
            bucket_name = 'rag-pdf-qa-bucket'  # Replace with your S3 bucket name
            object_name = uploaded_file.name
            success = upload_pdf_to_s3(uploaded_file, bucket_name, object_name)
            if success:
                st.sidebar.success("Uploaded PDF to S3 successfully!")

                # Initiate Textract job
                job_id = initiate_textract_job(bucket_name, object_name)
                if job_id:
                    st.sidebar.info("Textract job initiated. Extracting text...")
                    # Poll for Textract job completion
                    while True:
                        status = check_textract_job_status(job_id)
                        if status == 'SUCCEEDED':
                            extracted_text = get_textract_results(job_id)
                            st.session_state.pdf_text = extracted_text
                            st.sidebar.success("Text extraction completed.")
                            break
                        elif status == 'FAILED':
                            st.sidebar.error("Textract job failed.")
                            break
                        else:
                            time.sleep(5)  # Wait before polling again

                else:
                    st.sidebar.error("Failed to initiate Textract job.")

    # Option to input a PDF URL
    pdf_url = st.sidebar.text_input("Or enter a PDF URL:", placeholder="https://example.com/paper.pdf")
    if pdf_url:
        # Reset session if a new URL is entered
        if pdf_url not in st.session_state.uploaded_files:
            st.session_state.uploaded_files.append(pdf_url)
            st.session_state.questions = []
            st.session_state.answers = {}
            st.session_state.pdf_text = ""
            st.session_state.pdf_summary = ""
            st.session_state.topics = []

            # Download PDF from URL
            try:
                response = requests.get(pdf_url)
                response.raise_for_status()
                with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(response.content)
                    temp_file.flush()
                    temp_file_name = temp_file.name

                # Upload to S3
                bucket_name = 'rag-pdf-qa-bucket'  # Replace with your S3 bucket name
                object_name = os.path.basename(pdf_url)
                with open(temp_file_name, 'rb') as f:
                    success = upload_pdf_to_s3(f, bucket_name, object_name)
                if success:
                    st.sidebar.success("Uploaded PDF from URL to S3 successfully!")

                    # Initiate Textract job
                    job_id = initiate_textract_job(bucket_name, object_name)
                    if job_id:
                        st.sidebar.info("Textract job initiated. Extracting text...")
                        # Poll for Textract job completion
                        while True:
                            status = check_textract_job_status(job_id)
                            if status == 'SUCCEEDED':
                                extracted_text = get_textract_results(job_id)
                                st.session_state.pdf_text = extracted_text
                                st.sidebar.success("Text extraction completed.")
                                break
                            elif status == 'FAILED':
                                st.sidebar.error("Textract job failed.")
                                break
                            else:
                                time.sleep(5)  # Wait before polling again

                else:
                    st.sidebar.error("Failed to upload PDF to S3.")
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download PDF from URL {pdf_url}: {e}")
                st.sidebar.error(f"Failed to download PDF from URL: {e}")
            finally:
                if 'temp_file_name' in locals():
                    os.remove(temp_file_name)

    # Slider for number of questions
    num_questions = st.sidebar.slider(
        "Number of questions to generate:",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
        help="Select the number of questions you want to generate based on the document."
    )

    # Detect changes in slider value
    if num_questions != st.session_state.previous_num_questions:
        st.session_state.questions = []
        st.session_state.answers = {}
        st.session_state.previous_num_questions = num_questions
        st.session_state.pdf_summary = ""
        st.session_state.topics = []

    # Checkbox for summarization
    summarize = st.sidebar.checkbox("Summarize Document", value=False)
    st.session_state.summarize = summarize

    # Proceed only if PDF text is available
    if st.session_state.pdf_text:
        if st.session_state.summarize and not st.session_state.pdf_summary:
            with st.spinner("Summarizing the document..."):
                summary_prompt = (
                    "Summarize the following text in a concise manner:\n\n" + st.session_state.pdf_text
                )
                summary = generate_text_with_bedrock(summary_prompt)
                st.session_state.pdf_summary = summary
            st.write("## Summary of the Document")
            st.write(st.session_state.pdf_summary)
            text_for_topics = st.session_state.pdf_summary
        else:
            text_for_topics = st.session_state.pdf_text

        # Extract Topics
        if not st.session_state.topics:
            with st.spinner("Extracting key topics from the document..."):
                topics_prompt = (
                    "Extract the top 5 key topics from the following text:\n\n" + text_for_topics
                )
                topics_text = generate_text_with_bedrock(topics_prompt)
                # Assume topics are separated by new lines or commas
                topics = re.split(r'\n|,', topics_text)
                topics = [topic.strip('- ').strip() for topic in topics if topic.strip()]
                st.session_state.topics = topics[:5]
            # Optionally display topics
            # st.write("## Extracted Topics")
            # for idx, topic in enumerate(st.session_state.topics, 1):
            #     st.write(f"{idx}. {topic}")

        # Generate Questions
        if not st.session_state.questions:
            with st.spinner("Generating questions based on extracted topics..."):
                questions = []
                for topic in st.session_state.topics:
                    question_prompt = (
                        f"Generate a thought-provoking question related to the topic:\n\nTopic: {topic}\n\nQuestion:"
                    )
                    question = generate_text_with_bedrock(question_prompt)
                    if question:
                        questions.append(question)
                    if len(questions) >= num_questions:
                        break
                st.session_state.questions = questions
            if st.session_state.questions:
                st.success("Questions generated successfully!")
            else:
                st.warning("No questions were generated. Please try again.")

        # Display Generated Questions
        if st.session_state.questions:
            st.subheader("Generated Questions:")
            for idx, question in enumerate(st.session_state.questions, 1):
                if st.button(f"Q{idx}: {question}", key=f"question_{idx}"):
                    # Fetch or display the answer
                    if question not in st.session_state.answers:
                        with st.spinner("Fetching answer..."):
                            answer_prompt = (
                                f"Provide a comprehensive answer to the following question based on the document:\n\n"
                                f"Question: {question}\n\nAnswer:"
                            )
                            answer = generate_text_with_bedrock(answer_prompt)
                            st.session_state.answers[question] = answer
                    st.write(f"**Q{idx}:** {question}")
                    st.write(f"**Answer:** {st.session_state.answers[question]}")
            st.write("---")

        # Custom Query Input Field
        st.subheader("Ask a Custom Question")
        user_question = st.text_input("Enter your custom question here:", placeholder="Type your question and press Enter", key='custom_question_input')

        # Always display the 'Submit Custom Query' button
        if st.button("Submit Custom Query", key="submit_custom_query"):
            if not user_question.strip():
                st.error("Please enter a valid question.")
            else:
                with st.spinner("Fetching answer..."):
                    answer_prompt = (
                        f"Provide a comprehensive answer to the following question based on the document:\n\n"
                        f"Question: {user_question}\n\nAnswer:"
                    )
                    answer = generate_text_with_bedrock(answer_prompt)
                    st.session_state.answers[user_question] = answer
                st.write(f"**Your Question:** {user_question}")
                st.write(f"**Answer:** {st.session_state.answers[user_question]}")
                # Clear the input field
                st.session_state.custom_question_input = ""

    else:
        st.info("Please upload a PDF file or enter a PDF URL to generate questions.")

    # Reset Application Button
    if st.button("Reset Application", key="reset_application_button"):
        reset_session()

if __name__ == "__main__":
    main()
