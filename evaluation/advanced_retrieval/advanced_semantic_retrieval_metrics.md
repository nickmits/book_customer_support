# RAGAS Evaluation Metrics - Advanced Semantic Retrieval System

**Evaluation Date:** 2025-10-21 01:14:36

**Configuration:**
- Chunking Method: SemanticChunker
- Threshold Type: percentile
- Threshold Amount: 95
- Retrieval Components: Ensemble (BM25 + Multi-Query + Cohere Rerank)
- Model: gpt-4o-mini

## Metrics Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.7661 |
| Context Recall | 0.6000 |
| Context Precision | 0.6417 |
| Answer Relevancy | 0.7333 |
| Factual Correctness(Mode=F1) | 0.2100 |

**Average Score:** 0.5902

## Metric Descriptions

- **Faithfulness**: Measures factual consistency of the answer with the context
- **Context Recall**: Measures how well retrieved context aligns with ground truth
- **Context Precision**: Measures signal-to-noise ratio of retrieved contexts
- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Factual Correctness**: Measures factual overlap between answer and ground truth

*Score Range: 0.0 (worst) to 1.0 (best)*

## System Architecture

This evaluation uses the **Advanced Semantic Retrieval System** with:
- **SemanticChunker** for intelligent text splitting
- **Ensemble Retriever** combining BM25, Multi-Query, and Cohere Rerank
- **Qdrant** vector store for efficient similarity search
