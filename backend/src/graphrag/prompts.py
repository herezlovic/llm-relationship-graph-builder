"""Prompts for GraphRAG map-reduce global query-focused summarization."""

MAP_SYSTEM_PROMPT = """You are a helpful assistant answering questions using community summaries from a knowledge graph.
Use only the provided community summaries. If the summaries do not help answer the question, say so clearly.
Respond in exactly this format (no extra preamble):
helpfulness: <integer 0-100>
answer: <your intermediate answer>
"""

MAP_HUMAN_PROMPT = """Question:
{question}

Community summaries:
{community_summaries}

Produce a helpfulness score from 0-100 for how useful these summaries are for answering the question,
then write the intermediate answer grounded only in the summaries.
"""

REDUCE_SYSTEM_PROMPT = """You are a helpful assistant synthesizing a final global answer from ranked partial answers.
Use only the provided partial answers. Prefer higher-helpfulness answers. Be comprehensive and diverse
in perspectives covered, while remaining factual and concise.
"""

REDUCE_HUMAN_PROMPT = """Question:
{question}

Ranked partial answers (highest helpfulness first):
{partial_answers}

Write the final global answer for the user.
"""
