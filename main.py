import streamlit as st
import pandas as pd
from transformers import pipeline
import docx2txt
import PyPDF2
from io import BytesIO
import tempfile
import os

#page config.
st.set_page_config(
    page_title="Text Summarizer",
    page_icon="📝",
    layout="centered",
    initial_sidebar_state="collapsed"
)

#custom CSS for Apple-like style
st.markdown("""
<style>
    /*Import SF Pro font if available, fallback to system fonts*/
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /*Main container styling*/
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 3rem 2rem;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }

    /*Title styling*/
    .app-title {
        font-size: 3rem;
        font-weight: 300;
        color: #1d1d1f;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        font-size: 1.2rem;
        color: #86868b;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
    }

    /*File uploader styling*/
    .stFileUploader > div > div > div {
        background: #f5f5f7;
        border: 2px dashed #d2d2d7;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stFileUploader > div > div > div:hover {
        border-color: #007aff;
        background: #f0f7ff;
    }

    /*Button styling*/
    .stButton > button {
        background: linear-gradient(135deg, #007aff 0%, #5856d6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(0, 122, 255, 0.3);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 122, 255, 0.4);
    }

    /*Summary container*/
    .summary-container {
        background: #f5f5f7;
        border-radius: 16px;
        padding: 2rem;
        margin-top: 2rem;
        border-left: 4px solid #007aff;
    }

    .summary-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 1rem;
    }

    .summary-text {
        font-size: 1rem;
        line-height: 1.6;
        color: #424245;
        font-weight: 400;
    }

    /*Progress bar*/
    .stProgress .st-bo {
        background-color: #007aff;
    }

    /*Success/Error message*/
    .stSuccess, .stError {
        border-radius: 12px;
        border: none;
    }

    /*Hide Streamlit branding*/
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /*Responsive design*/
    @media (max-width: 768px) {
        .main-container {
            margin: 1rem;
            padding: 2rem 1.5rem;
        }

        .app-title {
            font-size: 2.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


#init. summarization pipeline
@st.cache_resource
def load_summarizer():
    try:
        summarizer = pipeline("summarization",
                              model="facebook/bart-large-cnn",
                              device=-1)  #Use CPU
        return summarizer
    except Exception as e:
        st.error(f"Error while loading model: {str(e)}")
        return None


def extract_text(uploaded_file):
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")

        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            #tempo. save of file uploaded
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            text = docx2txt.process(tmp_file_path)#text extraction

            #Clean up temporary file
            os.unlink(tmp_file_path)
            return text

        else:
            st.error("File format not supported! Please upload a TXT, PDF, or DOCX file.")
            return None

    except Exception as e:
        st.error(f"Error while reading file: {str(e)}")
        return None


def chunk_text(text, max_chunk_length=1024):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_length:
            current_chunk += sentence + '. '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def summarize_text(text, summarizer):
    try:
        # If text is short, return as is
        if len(text.split()) < 50:
            return text

        #Split long text into chunks
        chunks = chunk_text(text)
        summaries = []

        #Show progress
        progress_bar = st.progress(0)

        for i, chunk in enumerate(chunks):
            if len(chunk.split()) > 10:  #Only summarize chunks with meaningful content
                try:
                    summary = summarizer(chunk,
                                         max_length=150,
                                         min_length=30,
                                         do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except:
                    summaries.append(chunk[:200] + "...")  #fallback to truncation

            progress_bar.progress((i + 1) / len(chunks))

        progress_bar.empty()
        return " ".join(summaries)

    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None


def main():
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown('''
        <h1 class="app-title">Text Summarizer</h1>
        <p class="app-subtitle">Upload a document and get an intelligent summary</p>
    ''', unsafe_allow_html=True)

    summarizer = load_summarizer()

    if summarizer is None:
        st.error("Failed to load the summarization model. Please check your internet connection.")
        return

    uploaded_file = st.file_uploader(
        "Choose file",
        type=['txt', 'pdf', 'docx'],
        help="Supported formats: TXT, PDF, DOCX"
    )

    if uploaded_file is not None:
        st.success(f"File uploaded succesfuly: {uploaded_file.name} ({uploaded_file.size} bytes)")

        with st.spinner("Reading your document..."):
            text = extract_text(uploaded_file)

        if text:
            #text preview
            with st.expander("Document Preview", expanded=False):
                st.text_area("Content preview:", text[:500] + "..." if len(text) > 500 else text, height=150)

            if st.button("create Summary", type="primary"):
                with st.spinner("Creating your summary..."):
                    summary = summarize_text(text, summarizer)

                if summary:
                    st.markdown(f'''
                        <div class="summary-container">
                            <h3 class="summary-title">📋 Summary</h3>
                            <p class="summary-text">{summary}</p>
                        </div>
                    ''', unsafe_allow_html=True)

                    #sum stats.
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original Words", len(text.split()))
                    with col2:
                        st.metric("Summary Words", len(summary.split()))
                    with col3:
                        st.metric("Compression Ratio", f"{len(summary.split()) / len(text.split()):.1%}")

    else:
        st.markdown('''
            <div style="text-align: center; color: #86868b; margin-top: 2rem;">
                <p>👆 Upload a document to get started</p>
            </div>
        ''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()