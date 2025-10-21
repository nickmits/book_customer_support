# RAGAS Evaluation Metrics - Simple PDF Retriever

**Evaluation Date:** 2025-10-21 00:07:55

**Configuration:**
- Chunking Method: RecursiveCharacterTextSplitter
- Chunk Size: 1000
- Chunk Overlap: 200
- Model: gpt-4o-mini

## Metrics Results

| Metric | Score |
|--------|-------|
| Faithfulness | 0.9800 |
| Context Recall | 0.5000 |
| Context Precision | 0.2667 |
| Answer Relevancy | 0.9362 |
| Factual Correctness(Mode=F1) | 0.3820 |

**Average Score:** 0.6130

## Metric Descriptions

- **Faithfulness**: Measures factual consistency of the answer with the context
- **Context Recall**: Measures how well retrieved context aligns with ground truth
- **Context Precision**: Measures signal-to-noise ratio of retrieved contexts
- **Answer Relevancy**: Measures how relevant the answer is to the question
- **Factual Correctness**: Measures factual overlap between answer and ground truth

*Score Range: 0.0 (worst) to 1.0 (best)*
