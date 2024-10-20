# test_app.py

import streamlit as st

st.title("Test Streamlit App")

# File Uploader
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.success(f"Uploaded {len(uploaded_files)} file(s).")
    for uploaded_file in uploaded_files:
        st.write(f"**Filename:** {uploaded_file.name}")
        # Display first few bytes to confirm upload
        content = uploaded_file.read()
        st.write(f"**File size:** {len(content)} bytes")
else:
    st.info("Please upload at least one PDF file to proceed.")
