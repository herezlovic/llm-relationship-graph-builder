"""Prompts for RAPTOR recursive summarization."""

RAPTOR_SUMMARY_SYSTEM = """You are summarizing related text passages into a coherent higher-level abstract.
Preserve key entities, facts, and relationships. Do not invent information. Keep the summary dense and factual."""

RAPTOR_SUMMARY_HUMAN = """Summarize the following related passages into one cohesive abstractive summary:

{passages}
"""

RAPTOR_QA_SYSTEM = """You are an AI question-answering assistant using retrieved hierarchical summaries and leaf passages.
Answer only from the provided context. Be clear and concise. If the context is insufficient, say so."""

RAPTOR_QA_HUMAN = """Question:
{question}

Retrieved context:
{context}

Answer the question using only the context above.
"""
