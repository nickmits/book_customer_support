# RAGAS Evaluation Metrics - Simple Semantic PDF Retriever

**Evaluation Date:** 2025-10-21 00:35:00

**Configuration:**
- Chunking Method: SemanticChunker
- Threshold Type: percentile
- Threshold Amount: 95
- Model: gpt-4o-mini

## Metrics Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.9846 |
| Context Recall | 0.8000 |
| Context Precision | 0.8119 |
| Answer Relevancy | 0.9318 |
| Factual Correctness(Mode=F1) | 0.3860 |

**Average Score:** 0.7829

## Metric Descriptions

- **Faithfulness**: Measures factual consistency of the answer with the context
- **Context Recall**: Measures how well retrieved context aligns with ground truth
- **Context Precision**: Measures signal-to-noise ratio of retrieved contexts
- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Factual Correctness**: Measures factual overlap between answer and ground truth

*Score Range: 0.0 (worst) to 1.0 (best)*
