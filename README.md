# Generative Fashion Search System

## Objective

The aim of this project is to develop an intelligent fashion search system that enables users to search product descriptions and receive personalized recommendations based on their queries. The system leverages the Myntra dataset from Kaggle for its implementation.

## Project Scope

The project follows a Three-Layered Architecture to optimize search results and generate tailored product recommendations:

1. **Embedding Layer**: Processes and prepares product descriptions for embedding.
2. **Search Layer**: Performs semantic search using query embeddings and implements reranking for improved accuracy.
3. **Generation Layer**: Utilizes LLM prompts to generate personalized and accurate responses.

## Implementation Details

### 1. Embedding Layer

#### Step 1: Preprocess the Dataset
- **Objective**: Upload, clean, and preprocess the dataset containing product data.
- **Implementation**:
  - Load the Myntra dataset (CSV file).
  - Clean the product descriptions by removing irrelevant data, punctuation, and other noise.
  - Tokenize and process the text for further embedding generation.
  
#### Step 2: Generate Embeddings
- Use a pre-trained embedding model (such as Sentence-BERT) to generate embeddings for product descriptions.
- Store these embeddings for fast retrieval during the search phase.

### 2. Search Layer

#### Step 1: Perform Vector Search
- **Objective**: Execute a semantic search using query embeddings.
- **Implementation**:
  - Transform user queries into embeddings using the same pre-trained model.
  - Compare the query embeddings with product description embeddings using cosine similarity or other distance metrics.

#### Step 2: Re-Ranking
- **Objective**: Enhance search result accuracy by re-ranking results.
- **Implementation**:
  - Use HuggingFace cross-encoding models to re-rank the search results based on their relevance to the user’s query.

#### Step 3: Caching Mechanism
- **Objective**: Optimize the performance of the search layer.
- **Implementation**:
  - Implement a caching mechanism to store previously computed results, reducing redundant computations for frequently searched queries.

### 3. Generation Layer

#### Step 1: Design the Prompt for the LLM
- **Objective**: Tailor the prompts to generate accurate and meaningful product recommendations.
- **Implementation**:
  - Design detailed prompts that combine user queries with the retrieved search results.
  - Use a Large Language Model (LLM) to generate personalized responses that offer users the most relevant product recommendations based on their needs.

## Requirements

- Python 3.x
- Libraries:
  - HuggingFace Transformers
  - Sentence-BERT
  - Pandas
  - Scikit-learn
  - NumPy
  - PyTorch
  - Flask (for deploying the API)

