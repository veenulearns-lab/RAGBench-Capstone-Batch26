
## CP-2 VectorDB + Multi-LLM (For Sridhar)

Part A compares FAISS vs Chroma retrieval across all 5 domains.
Part B compares llama_8b / llama_70b / qwen3_27b as the generator, FAISS-only, across all 5 domains.

| File | Examples | Rel RMSE↓ | Comp RMSE↓ | Adh AUCROC↑ |
|---|---|---|---|---|
| cp2_llm_llama_70b_covidqa_final.csv | 50 | 0.2799 | 0.7102 | 0.5474 |
| cp2_llm_llama_70b_cuad_final.csv | 50 | 0.4169 | 0.8021 | 0.6064 |
| cp2_llm_llama_70b_finqa_final.csv | 49 | 0.5224 | 0.8921 | 0.5023 |
| cp2_llm_llama_70b_hotpotqa_final.csv | 50 | 0.1796 | 0.7442 | 0.7727 |
| cp2_llm_llama_70b_techqa_final.csv | 50 | 0.3465 | 0.6625 | 0.5357 |
| cp2_llm_llama_8b_covidqa_final.csv | 50 | 0.3784 | 0.6960 | 0.6463 |
| cp2_llm_llama_8b_cuad_final.csv | 48 | 0.5884 | 0.7926 | 0.4333 |
| cp2_llm_llama_8b_finqa_final.csv | 49 | 0.6056 | 0.8620 | 0.5556 |
| cp2_llm_llama_8b_hotpotqa_final.csv | 50 | 0.1831 | 0.7442 | 0.8068 |
| cp2_llm_llama_8b_techqa_final.csv | 48 | 0.4196 | 0.7100 | 0.5000 |
| cp2_llm_qwen3_27b_covidqa_final.csv | 48 | 0.5381 | 0.6969 | 0.5513 |
| cp2_llm_qwen3_27b_cuad_final.csv | 47 | 1.2344 | 0.7451 | 0.6023 |
| cp2_llm_qwen3_27b_finqa_final.csv | 50 | 0.7892 | 0.7817 | 0.5222 |
| cp2_llm_qwen3_27b_hotpotqa_final.csv | 50 | 0.5893 | 0.7487 | 0.5795 |
| cp2_llm_qwen3_27b_techqa_final.csv | 46 | 0.3200 | 0.6395 | 0.5000 |
| cp2_vdb_chroma_covidqa_final.csv | 50 | 0.3807 | — | 0.6463 |
| cp2_vdb_chroma_cuad_final.csv | 48 | 0.5724 | — | 0.4333 |
| cp2_vdb_chroma_finqa_final.csv | 49 | 0.6014 | — | 0.5444 |
| cp2_vdb_chroma_hotpotqa_final.csv | 50 | 0.1862 | — | 0.7235 |
| cp2_vdb_chroma_techqa_final.csv | 48 | 0.4267 | — | 0.5000 |
| cp2_vdb_faiss_covidqa_final.csv | 50 | 0.3792 | — | 0.6463 |
| cp2_vdb_faiss_cuad_final.csv | 48 | 0.5861 | — | 0.6000 |
| cp2_vdb_faiss_finqa_final.csv | 49 | 0.5983 | — | 0.5556 |
| cp2_vdb_faiss_hotpotqa_final.csv | 50 | 0.1877 | — | 0.7348 |
| cp2_vdb_faiss_techqa_final.csv | 48 | 0.4220 | — | 0.5000 |
