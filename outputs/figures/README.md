# Wildlife1 reproduction figures

Generated from the completed Project-Ava run.

- Paper main AVA-100 result: 75.8% on 120 questions, using
  Qwen2.5-32B for SA and Gemini-1.5-Pro for CA.
- Our wildlife1 SA result: 75.0% on 8 questions, using Qwen2.5-14B.
- Our wildlife1 CA result with 4 votes: 37.5%.
- Our wildlife1 CA result with 8 votes: 50.0%.
- Our CA model: Qwen2.5-VL-7B, with at most 128 uniformly sampled frames.

The paper and reproduction bars have different model configurations and sample
sizes. The comparison is contextual, not a claim of exact statistical replication.

Files:
- accuracy_comparison.png/pdf
- per_question_correctness.png/pdf
- ca_vote_ablation.png/pdf
- knowledge_graph_local.png/pdf
- wildlife1_results.csv
- knowledge_graph_local_nodes.csv
- knowledge_graph_stats.json
