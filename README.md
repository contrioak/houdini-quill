## Project: Houdini 🪄 - Your Magical Text & PDF AI Assistant

Here's Houdini, an awesome AI assistant that can answer your questions in two ways:

**1. PDF Genie ‍ :** 

   - Upload and process your PDFs 
   - Ask questions about what's inside the PDFs and get insightful answers 

**2. Multi-Search RAG Agent   (I've exausted the OPENAI API credits!!)** :**

   - Search across amazing sources like Wikipedia , Arxiv , and Langsmith docs  
   - Get answers to your questions from these incredible resources   

### How to Unleash Houdini's Magic ✨ :

1. **Setting Up:**

   - You'll need some cool Python libraries like `streamlit`, `langchain`, and `PyPDF2`. Install them with `pip install streamlit langchain PyPDF2`.
   - For the PDF Genie, you'll also need a Google Cloud AI Platform key ([https://ai.google.dev/gemini-api/docs/api-key])).  

2. **Running the Project:**

   - Save the provided code as a Python file (e.g., `app.py`).
   - Create a virtual environment using `python3 -m venv venv`
   - Activate the environment
   - Open a terminal, navigate to the file's location, and run:
     ```bash
     streamlit run app.py
     ```
   - This will launch a Streamlit web app in your browser, ready for your commands!

3. **Using the PDF Genie:**

   - In "File Upload & Processing", upload your PDFs. Click "Upload & Process" to create a magic index for searching 🪄.
   - Once processing is complete, ask your questions about the PDFs in the text box. Houdini will analyze them and answer in detail!

4. **Using the Multi-Search RAG Agent (Coming Soon!)** :

   - Enter your query in the text box under "Multi-Search RAG Agent". 
   - **Note:** Due to limitations with the OpenAI API, this feature is currently under construction ️. Stay tuned for its epic arrival!

### Features  :

- **PDF Processing & Question Answering:** Houdini can process your PDFs and answer your questions about their content, just like magic 🪄
- **Multi-Source Search (Coming Soon!)** : Search across Wikipedia, Arxiv, and Langsmith documentation for answers.
- **Detailed Responses:** Houdini strives to provide comprehensive and informative answers to your questions, like a wise old owl .
- **User-friendly Interface:** The Streamlit web app makes interacting with Houdini easy and intuitive, just like a wave of your wand 🪄.


Houdini demonstrates the power of AI for text processing and information retrieval. With further development, Houdini can become even more versatile and helpful for various tasks!
