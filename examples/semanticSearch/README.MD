Endee Semantic Search Demo:

1. Project OverView:
     This project demonstrates an AI-powered semantic search system that retrieves relevant information from documents based on meaning rather than exact keyword matching.
     Traditional keyword-based search engines often fail when the query does not contain the exact words present in the documents. Semantic search solves this problem by converting text into vector embeddings, allowing the system to understand the context and meaning of the query.
     This application allows users to input a natural language question and retrieve the most relevant document chunks using vector similarity search.

2. Problem Statement:
     Many information retrieval systems rely on keyword matching, which often leads to inaccurate results when users phrase their queries differently from the stored text.
     For example:
       Query: What is machine learning?
       Document: Machine learning is a field of artificial intelligence where algorithms learn patterns from data.
     A keyword-based search might miss this result if the exact words don't match.
     This project solves that problem using semantic search with vector embeddings, enabling the system to retrieve results based on meaning and contextual similarity.

3. System Design and Technical Approach:
     The system follows a vector-based semantic search pipeline.
     Architecture:
                      Documents
                          ↓
                    Text Chunking
                          ↓
           Sentence Transformer Embeddings
                          ↓
               Vector Representation
                          ↓
                     User Query
                          ↓
                   Query Embedding
                          ↓
               Cosine Similarity Search
                          ↓
             Retrieve Top Relevant Results
                          ↓
              Generate Context-Based Response

     Key Components:
      Document Processing - Text documents are loaded and split into smaller chunks.
      Embedding Generation - Each chunk is converted into a vector embedding using the Sentence Transformer model (all-MiniLM-L6-v2).
      Similarity Search - When a user enters a query, it is converted into an embedding , Cosine similarity is used to compare the query vector with stored       document vectors.
      Result Retrieval - The system retrieves the top relevant document chunks.
      Context-Based Answer Generation - Retrieved chunks are combined to generate a contextual answer for the user

4. Explanation of How Endee is Used:
     This project is built on top of the Endee repository, which provides infrastructure for building high-performance vector-based AI systems.
     Endee is designed to support vector storage and retrieval for AI workloads, such as: Semantic search, Retrieval-Augmented Generation (RAG), Recommendation systems
     In this project:
       The Endee repository serves as the base environment for building the AI application.
       The semantic search example is implemented inside the repository under: examples/ragDemo/
     The example demonstrates how vector embeddings and similarity search can be used as the core retrieval mechanism in AI systems that could later integrate with a vector database like Endee.

5. Setup and Execution Instructions:
     Clone the Repository - 
       git clone https://github.com/aashima2693/endee.git
       cd endee/examples/ragDemo

     Install Dependencies -
       pip install -r requirements.txt
     
     Generate Document Embeddings -
      Run the embedding script to convert documents into vector embeddings :  python embed.py
      This will create: embeddings.json
    
     Run Semantic Search (CLI) - 
      You can test the search system from the terminal: python query.py
      Example query: What is machine learning?
     
     Run the Web Application - 
       Launch the Streamlit interface: streamlit run app.py

6. Example Queries:
     Try asking questions such as:
       What is machine learning?
       Explain artificial intelligence
       Why is Python popular?

7. Technologies Used:
     Python
     Streamlit
     Sentence Transformers
     NumPyScikit-learn

    
    