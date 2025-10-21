# RAGAS Evaluation Metrics - Advanced Recursive Retrieval System


**Configuration:**
- Chunking Method: RecursiveCharacterTextSplitter
- Chunk Size: 1000
- Chunk Overlap: 200
- Retrieval Components: Ensemble (BM25 + Multi-Query + Cohere Rerank)
- Model: gpt-4o-mini

## Metrics Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.9647 |
| Context Recall | 0.8333 |
| Context Precision | 0.5689 |
| Answer Relevancy | 0.8436 |
| Factual Correctness(Mode=F1) | 0.2571 |

**Average Score:** 0.6935

## Metric Descriptions

- **Faithfulness**: Measures factual consistency of the answer with the context
- **Context Recall**: Measures how well retrieved context aligns with ground truth
- **Context Precision**: Measures signal-to-noise ratio of retrieved contexts
- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Factual Correctness**: Measures factual overlap between answer and ground truth

*Score Range: 0.0 (worst) to 1.0 (best)*

## System Architecture

This evaluation uses the **Advanced Recursive Retrieval System** with:
- **RecursiveCharacterTextSplitter** for fast, efficient chunking
- **Ensemble Retriever** combining BM25, Multi-Query, and Cohere Rerank
- **Qdrant** vector store for efficient similarity search
