Okay, I understand. Here is the markdown content for your README file, without the Canvas formatting:

# PDF Q\&A System with Google Gemini and LangChain

This project demonstrates a Question-Answering (Q\&A) system that can answer questions based on the content of a PDF document. It leverages Google's Gemini Pro model for language understanding and generation, and LangChain for orchestrating the document loading, splitting, embedding, and retrieval processes.

## Features

  * **PDF Document Loading**: Loads content from a specified PDF file.

  * **Text Splitting**: Divides the document into manageable chunks for processing.

  * **Text Cleaning**: Removes emojis and other symbols from the text.

  * **Vector Embedding**: Converts text chunks into numerical vector representations using Google Generative AI Embeddings.

  * **Vector Storage**: Stores the embedded vectors in a ChromaDB for efficient similarity search.

  * **Retrieval-Augmented Generation (RAG)**: Retrieves relevant document chunks based on a user's query and uses them to inform the Gemini model's answer.

  * **Interactive Q\&A**: Allows users to ask questions and get answers from the PDF content.

## Installation

To set up the project, first install the necessary Python libraries:

```bash
pip install -q --upgrade google-generativeai langchain-google-genai chromadb pypdf python-dotenv
```

## Setup

1.  **Google Gemini API Key**:
      * Obtain a Google Gemini API key from the [Google AI Studio](https://aistudio.google.com/app/apikey).
      * Create a file named `.env` in the root directory of your project.
      * Add your API key to the `.env` file in the following format:
        ```
        Gemini_key="YOUR_API_KEY_HERE"
        ```
        Replace `"YOUR_API_KEY_HERE"` with your actual Gemini API key.

## Usage

The provided code is a script that performs the following steps:

1.  **Imports Libraries**: Imports all necessary modules.

2.  **Loads API Key and Models**: Initializes the Gemini Pro model for chat and the embedding model using your API key.

3.  **Loads and Splits Document**:

      * It expects a PDF file named `Practical Statistics for Data Scientists.pdf` inside a `Documents` folder in the same directory as your script.
      * It loads this PDF and splits its content into smaller, overlapping text chunks.
      * It then cleans these text chunks by removing emojis and symbols.

4.  **Creates Vector Index**: It generates embeddings for the text chunks and stores them in a ChromaDB instance located in a `./Database` directory.

5.  **Sets up QA Chain**: It configures a RetrievalQA chain that uses the Gemini model and the vector index to answer questions.

6.  **Asks a Question**: It demonstrates how to ask a question to the system and prints the answer.

To run the Q\&A system:

1.  Make sure you have a `Documents` folder with `Practical Statistics for Data Scientists.pdf` inside it.

2.  Ensure your `.env` file with the `Gemini_key` is correctly set up.

3.  Execute the Python script. The script will load the document, process it, and then ask a predefined question, printing the answer to the console.

<!-- end list -->

```python
# Example of asking a question
question="What is the difference between a histogram and a bar chart?"
ans=ask_question(question,qa_chain)
print(ans)
```

You can modify the `question` variable in the `ask_question` function call to ask different questions about the PDF content.

## Code Structure

The script is organized into several functions, each responsible for a specific part of the Q\&A pipeline:

  * **`load_model()`**: Initializes the `ChatGoogleGenerativeAI` and `GoogleGenerativeAIEmbeddings` models using the API key.

  * **`load_document(document_path)`**: Loads a PDF document from the given path using `PyPDFLoader` and returns its pages.

  * **`split_text(pages)`**: Splits the loaded document pages into text chunks using `RecursiveCharacterTextSplitter`.

  * **`remove_emojis(string)`**: A utility function to clean text by removing emojis.

  * **`vector_index(texts, embeddings, persist_directory)`**: Creates and persists a ChromaDB vector store from the text chunks and embeddings.

  * **`qa_chain(vector_index, model)`**: Sets up the `RetrievalQA` chain for question answering, connecting the language model with the retriever.

  * **`ask_question(question, qa_chain)`**: Takes a question and the QA chain, processes the query, and returns the answer.


https://github.com/waleedgul92/DataBot-Data-Answer-Tool/assets/84980384/ff4d85db-d15b-4ab3-8a7d-731eb3f99f6f

