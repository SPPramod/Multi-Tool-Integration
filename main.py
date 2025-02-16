import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import langchain_google_genai
import pandas as pd
import plotly.express as px
import requests
import json

API_KEY = st.secrets["API_KEY"] 
genai.configure(api_key=API_KEY)

if 'pdf_chat_history' not in st.session_state:
    st.session_state['pdf_chat_history'] = []

if 'pdf_chat_mode' not in st.session_state:
    st.session_state['pdf_chat_mode'] = False

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

st.set_page_config(page_title="Multi Tool Integration", layout="wide")

def get_gemini_response(question):
    model = genai.GenerativeModel("gemini-pro")
    chat = model.start_chat(history=[])
    response = chat.send_message(question, stream=True)
    return response

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = langchain_google_genai.GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=API_KEY
    )
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, api_key=API_KEY)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def generate_gemini_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

try:
    st.title("Multi Tool Integration For Research Assistance")

    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Go to", ["Home", "ChatBot", "Image Captioning", "PDF Reader", "Data Analysis", "Knowledge Graph"])

    if page == "Home":
        st.write("""
        Multi-tool integration for research assistance leverages various AI-powered tools to streamline and enhance the research process. This comprehensive approach combines different technological capabilities to provide a more efficient and thorough research experience.
        
        The research assistance toolkit includes the following key features:
        
        Intelligent Chat Interface: An AI-powered chatbot that helps answer questions, explain concepts, and provide detailed insights across various topics.
        Document Analysis: Advanced PDF processing capabilities that allow for text extraction, summarization, and interactive Q&A with document contents.
        Visual Content Processing: Image analysis and captioning features that help interpret and describe visual information in research materials.
        Data Visualization: Tools for analyzing and visualizing datasets, helping researchers identify patterns and trends in their data.
        Knowledge Graph Integration: Access to structured knowledge bases that provide detailed information about various topics and their relationships.
        """)
        images = ["img1.jpg", "img2.jpeg", "img3.jpeg"]
        cols = st.columns(3)
        
        for idx, image_path in enumerate(images):
            with cols[idx % 3]:
                st.image(image_path, use_container_width=True)

    elif page == "ChatBot":
        st.title("ChatBot Service")
        user_input = st.text_input("Input:", key="input")
        submit = st.button("Ask the Question")
        
        if submit and user_input:
            response = get_gemini_response(user_input)
            st.session_state['chat_history'].append(("You", user_input))
            st.subheader("Response")
            for chunk in response:
                st.write(chunk.text)
                st.session_state['chat_history'].append(("Bot", chunk.text))

        st.subheader("Chat History")
        for role, text in st.session_state['chat_history']:
            st.write(f"{role}: {text}")

    elif page == "Image Captioning":
        st.title("Generate Caption with Hashtags")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None and st.button('Upload'):
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                img = Image.open(uploaded_file)
                caption = model.generate_content(["Generate a detailed caption that accurately describes the content, mood, and potential story of the image in english", img])
                tags = model.generate_content(["Generate 10 trending hashtags for the image in a line in English", img])
                st.image(img, caption=f"Caption: {caption.text}")
                st.write(f"Tags: {tags.text}")
            except Exception as e:
                st.error(f"Failed to generate caption due to: {str(e)}")

    elif page == "PDF Reader":
        st.header("PDF Reader")
        uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        
        if uploaded_files:
            text = get_pdf_text(uploaded_files)
            text_chunks = get_text_chunks(text)
            get_vector_store(text_chunks)
            st.write("PDF text and vector store created successfully!")

            col1, col2 = st.columns(2)
            with col1:
                summarize_button = st.button("SUMMARIZE")
            with col2:
                chat_button = st.button("CHAT")

            if summarize_button:
                with st.spinner('Summarizing...'):
                    summary = generate_gemini_content(text, "Provide a detailed summary of the following text, ensuring all key points, arguments, and supporting details are included. Maintain the original text's structure and flow as much as possible")
                    if summary:
                        st.subheader("Summary")
                        st.write(summary)

            if chat_button:
                st.session_state['pdf_chat_mode'] = True

            if st.session_state['pdf_chat_mode']:
                st.subheader("PDF QnA Chat")
                
                with st.form("pdf_chat_form", clear_on_submit=True):
                    question = st.text_input("Ask a question about the PDF:")
                    submit_question = st.form_submit_button("Send")

                if submit_question and question:
                    try:
                        chain = get_conversational_chain()
                        vector_store = FAISS.load_local(
                            "faiss_index",
                            embeddings=langchain_google_genai.GoogleGenerativeAIEmbeddings(
                                model="models/embedding-001", 
                                google_api_key=API_KEY
                            ),
                            allow_dangerous_deserialization=True
                        )
                        docs = vector_store.similarity_search(question)
                        answer = chain.run(input_documents=docs, question=question)

                        st.session_state['pdf_chat_history'].append(("You", question))
                        st.session_state['pdf_chat_history'].append(("Bot", answer))

                        st.subheader("Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    elif page == "Data Analysis":
        st.title("Data Analysis and Visualization")
        
        uploaded_file = st.file_uploader("Upload CSV file for Analysis", type=["csv"])
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.subheader("Data Preview")
            st.dataframe(df.head())

            st.subheader("Data Summary")
            st.write(df.describe())

            st.subheader("Correlation Matrix")
            numeric_df = df.select_dtypes(include='number')

            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                fig_corr = px.imshow(corr_matrix, 
                                    text_auto=True, 
                                    title="Correlation Matrix",
                                    color_continuous_scale='Viridis')
                st.plotly_chart(fig_corr)
            else:
                st.warning("No numeric columns available for correlation matrix.")

            st.subheader("Visualizations")
            chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Histogram", "Scatter Plot"])
            columns = df.columns.tolist()

            if chart_type == "Line Chart":
                x_axis = st.selectbox("X-Axis", options=columns)
                y_axis = st.selectbox("Y-Axis", options=columns)
                line_chart = px.line(df, x=x_axis, y=y_axis, title=f"Line Chart: {y_axis} vs {x_axis}")
                st.plotly_chart(line_chart)

            elif chart_type == "Bar Chart":
                x_axis = st.selectbox("X-Axis", options=columns)
                y_axis = st.selectbox("Y-Axis", options=columns)
                bar_chart = px.bar(df, x=x_axis, y=y_axis, title=f"Bar Chart: {y_axis} vs {x_axis}")
                st.plotly_chart(bar_chart)

            elif chart_type == "Histogram":
                column = st.selectbox("Select Column", options=columns)
                histogram = px.histogram(df, x=column, title=f"Histogram: {column}")
                st.plotly_chart(histogram)

            elif chart_type == "Scatter Plot":
                x_axis = st.selectbox("X-Axis", options=columns)
                y_axis = st.selectbox("Y-Axis", options=columns)
                scatter_plot = px.scatter(df, x=x_axis, y=y_axis, title=f"Scatter Plot: {y_axis} vs {x_axis}")
                st.plotly_chart(scatter_plot)

    elif page == "Knowledge Graph":
        st.title("Knowledge Graph Search")
        query = st.text_input("Enter a keyword to search in Knowledge Graph:")

        if st.button("Search Knowledge Graph"):
            url = "https://kgsearch.googleapis.com/v1/entities:search"
            params = {
                'query': query,
                'key': API_KEY,
                'limit': 5,
                'indent': True
            }

            response = requests.get(url, params=params)
            data = response.json()

            filtered_results = []
            for item in data.get('itemListElement', []):
                result = item.get('result', {})
                if 'Person' in result.get('@type', []):
                    filtered_results.append(result)

            if filtered_results:
                for result in filtered_results:
                    name = result.get("name", "No Name")
                    description = result.get("description", "No Description")
                    detailed_desc = result.get("detailedDescription", {}).get("articleBody", "No Detailed Description")
                    url = result.get("detailedDescription", {}).get("url", "")

                    st.subheader(name)
                    st.write(f"**Description:** {description}")
                    st.write(f"**Detailed Description:** {detailed_desc}")
                    if url:
                        st.markdown(f"[Read more]({url})")
            else:
                st.warning("No relevant results found.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
