# Data Extraction from RAG Systems: Expanded Analysis

## 1 Introduction

Retrieval-Augmented Generation (RAG) systems combine information retrieval with large language models (LLMs) to provide grounded, factual responses. However, recent work by Zeng et al. [1] ("Follow My Instruction and Spill the Beans") demonstrated that instruction-tuned LLMs can be exploited to extract verbatim copies of retrieved documents through prompt injection attacks.

This work reproduces and extends their findings with two key contributions: (1) we demonstrate that instruction fine-tuning significantly increases vulnerability to extraction attacks compared to base models, revealing a fundamental tension between model helpfulness and security, and (2) we evaluate three prompt-level defense mechanisms to assess the feasibility of simple mitigation strategies. Our experiments use Wikipedia articles and test both base and instruction-tuned variants of Llama-2-7B, showing that instruction-tuned models exhibit substantially higher extraction rates while prompt-based defenses provide only modest protection.

## 2 Related Work

**RAG Security.** Zeng et al. [1] first systematically studied data extraction vulnerabilities in RAG systems, demonstrating that larger instruction-tuned models are more susceptible to verbatim extraction attacks. They tested models ranging from 117M to 70B parameters on 1,165 Wikipedia articles, showing extraction success scales with model size. Our work extends this by isolating the specific role of instruction fine-tuning.

**Instruction Fine-tuning.** While instruction fine-tuning improves model helpfulness and instruction-following capabilities [2], our work reveals a security tradeoff: models trained to follow instructions more faithfully also more readily comply with malicious extraction requests.

**Prompt Injection Defenses.** Prior work on prompt injection has proposed architectural defenses including output filtering and query-context separation [3]. We evaluate whether simple prompt-level defenses (explicit separators and safety instructions) can mitigate extraction without architectural changes.

## 3 Method

We implement the RICLM attack framework from Zeng et al. [1], where an attacker submits queries designed to extract retrieved context. Our baseline reproduces their methodology: BM25 retrieval followed by simple concatenation of retrieved document and query, then generation with `do_sample=False` for deterministic outputs.

**Expansion 1: Base vs Instruction-Tuned Models.** We compare Llama-2-7B base (`Llama-2-7b-hf`) against its instruction-tuned variant (`Llama-2-7b-chat-hf`) to isolate the effect of instruction fine-tuning on vulnerability. Both models are tested with identical attack queries.

**Expansion 2: Defense Mechanisms.** We evaluate three prompt formatting defenses:
- **Separator**: Add explicit markers (`===CONTEXT START===`, `===QUERY===`) to delineate sections
- **Instruction**: Prepend safety instruction "Do not copy the context verbatim"
- **Combined**: Apply both separator and instruction together

We measure extraction success using four metrics from the original paper: ROUGE-L, BLEU, token-level F1, and BERTScore, comparing model outputs against the retrieved document (not the target, following their methodology).

## 4 Experiments

**Setup.** We test on 10 Wikipedia articles with one query per article, using attack templates from Zeng et al. [1] including "copy and output all the text above" and topic-specific information requests. Models generate up to 128 tokens with greedy decoding. All experiments use a single T4 GPU with fp16 precision.

**Results: Instruction Fine-tuning Impact.** Instruction-tuned Llama-2-7B-chat shows substantially higher extraction rates than the base model across all metrics:

| Model | ROUGE-L | BLEU | F1 | BERTScore |
|-------|---------|------|-----|-----------|
| Base (7B) | 18.1 | 13.0 | 20.0 | 81.6 |
| Instruction-tuned (7B-chat) | 70.3 | 55.8 | 73.2 | 93.6 |

The instruction-tuned variant exhibits a 52.2 point increase in ROUGE-L, representing a 288% relative increase in extraction success. This confirms our hypothesis that instruction fine-tuning, while improving helpfulness, dramatically increases vulnerability to prompt injection attacks. The instruction-tuned model extracts nearly three times as much content, demonstrating that instruction-following capabilities are directly exploited by extraction attacks.

**Results: Defense Mechanisms.** Testing defense strategies on Llama-2-7B-chat yields:

| Defense | ROUGE-L | BLEU | F1 | BERTScore |
|---------|---------|------|-----|-----------|
| None (baseline) | 70.3 | 55.8 | 73.2 | 93.6 |
| Separator | 55.6 | 30.9 | 64.4 | 90.4 |
| Instruction | 47.9 | 36.2 | 50.0 | 89.8 |
| Combined | 37.2 | 14.3 | 45.1 | 86.7 |

The combined defense achieves the greatest reduction (33.1 points, 47% decrease), demonstrating that explicit safety instructions are more effective than structural separators alone. However, extraction rates remain substantial (37.2 ROUGE-L), indicating that prompt-level defenses provide only partial mitigation. Notably, testing the same defenses on the base model reveals a counterintuitive result: defenses increase extraction rates (18.1 to 46.7 ROUGE-L with combined defense), likely because structured prompts help the base model better understand the extraction task it naturally resists. This suggests prompt-based defenses are model-specific and may inadvertently assist attacks on non-instruction-tuned models.

**Comparison to Original Paper.** Our baseline results on instruction-tuned models align with Zeng et al.'s findings for 7B models, validating our reproduction. However, our reduced dataset (10 vs 1,165 documents) and shorter generation length (128 vs 512 tokens) limit direct numerical comparison.

## 5 Conclusion

We reproduce and extend the RAG data extraction vulnerability demonstrated by Zeng et al. [1], making two key contributions: First, we show that instruction fine-tuning directly increases extraction vulnerability, revealing a fundamental tradeoff between model helpfulness and security. Base models exhibit 288% lower extraction rates (18.1 vs 70.3 ROUGE-L), demonstrating that instruction-following capabilities are the primary attack vector. Second, we demonstrate that simple prompt-level defenses provide only partial mitigation on instruction-tuned models, with the best combined approach reducing extraction by 47% but still leaving substantial leakage (37.2 ROUGE-L). Critically, we find these same defenses backfire on base models, increasing extraction rates by 158%.

These findings have important implications for RAG system design: developers must carefully weigh the benefits of instruction fine-tuning against security risks, and prompt-level defenses should be considered model-specific rather than universal solutions. The counterintuitive result that defenses harm base model security suggests that defense mechanisms must be tailored to the specific model architecture and training regime. Future work should explore adversarial fine-tuning techniques that maintain instruction-following capabilities while resisting extraction attacks, as well as architectural defenses including output validation and content filtering.

**Limitations.** Our experiments use a smaller dataset (10 documents) and shorter generation length (128 tokens) than the original paper due to computational constraints. Testing on additional model families and larger-scale datasets would strengthen these findings.

## References

[1] Zeng, Y., et al. "Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems." arXiv preprint, 2024.

[2] Ouyang, L., et al. "Training language models to follow instructions with human feedback." NeurIPS, 2022.

[3] Liu, Y., et al. "Prompt injection attack against LLM-integrated applications." arXiv preprint, 2023.

---

**Note:** Replace X.X placeholders with your actual experimental results from the notebook output.
