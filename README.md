# Text-Summarizer

A simple AI-powered web application for automatic text summarization of documents in multiple formats, implemented with Streamlit.

## Repository Contents

- **app.py** – Core application with file processing, AI summarization, and user interface
- **README.md** – Project documentation and setup instructions

## Description

This project explores natural language processing for document summarization. The goal is to process various file formats, apply transformer-based AI models and generate intelligent text summaries. It serves as a practical application of NLP and web deployment.

## Features

- Multi-format document support (TXT, PDF, DOCX)
- AI-powered abstractive summarization using BART-large-cnn model
- Elegant, Apple-inspired user interface
- Text chunking for processing long documents
- Real-time statistics and compression ratios

## Technology Stack

The application uses:
- **Streamlit** for web interface
- **Hugging Face Transformers** for AI summarization
- **PyPDF2 & docx2txt** for document text extraction
- **Custom CSS** for Apple-style design aesthetics

## Deployment
*Simple deployment: `streamlit run app.py`*
