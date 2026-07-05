from __future__ import annotations

import os
import re
from typing import Any

try:
    from .retrieve import tokenize
except ImportError:
    from retrieve import tokenize

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "for",
    "how",
    "in",
    "is",
    "of",
    "on",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}

DEFINITION_PATTERNS = (
    " is ",
    " are ",
    " refers to ",
    " means ",
    " is defined as ",
)


def clean_text(text: str) -> str:
    """
    Remove extra spaces and normalize text formatting.
    """
    return re.sub(r"\s+", " ", text).strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Break large text into readable sentences.
    Very short fragments are ignored.
    """
    raw_sentences = re.split(r"(?<=[.!?])\s+", text)

    cleaned_sentences = [
        clean_text(sentence)
        for sentence in raw_sentences
    ]

    return [
        sentence
        for sentence in cleaned_sentences
        if len(sentence.split()) >= 6
    ]


def remove_extension(filename: str) -> str:
    """
    Return filename without extension.
    """
    return filename.rsplit(".", 1)[0]


def build_source_label(chunk: dict[str, Any], index: int) -> str:
    """
    Create a readable source reference for output.
    """
    source_name = str(chunk.get("source") or "document")

    title = str(chunk.get("title") or "").strip()

    metadata_parts: list[str] = []

    if chunk.get("page"):
        metadata_parts.append(f"page {chunk['page']}")

    if chunk.get("section_id") is not None:
        metadata_parts.append(f"section {chunk['section_id']}")

    extra_info = (
        f" ({', '.join(metadata_parts)})"
        if metadata_parts
        else ""
    )

    if title and title != remove_extension(source_name):
        return f"[{index}] {title} - {source_name}{extra_info}"

    return f"[{index}] {source_name}{extra_info}"


def normalize_context(
    context: list[str] | list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Ensure all retrieved chunks follow the same structure.
    """
    normalized_chunks: list[dict[str, Any]] = []

    for item in context:
        if isinstance(item, dict):
            normalized_chunks.append(item)
        else:
            normalized_chunks.append(
                {
                    "text": item,
                    "source": "retrieved context",
                    "score": 0.0,
                }
            )

    return normalized_chunks


def try_transformer_generation(
    query: str,
    context_text: str,
) -> str | None:
    """
    Try generating an answer using a local transformer model.
    Falls back gracefully if the model is unavailable.
    """
    model_name = os.getenv("RAG_GENERATION_MODEL")

    if not model_name:
        return None

    try:
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

    except ImportError:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            local_files_only=True,
        )

        prompt = (
            "Answer the question using only the provided context.\n\n"
            f"Question: {query}\n\n"
            f"Context: {context_text}\n\n"
            "Answer:"
        )

        encoded_input = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )

        generated_output = model.generate(
            **encoded_input,
            max_new_tokens=180,
        )

        final_answer = tokenizer.decode(
            generated_output[0],
            skip_special_tokens=True,
        ).strip()

        return final_answer or None

    except Exception:
        return None


def extract_definition_subject(query: str) -> str | None:
    """
    Extract the main subject from definition-style questions.
    Example:
    'What is machine learning?' -> 'machine learning'
    """
    match = re.search(
        r"\b(?:what|who)\s+(?:is|are|was|were)\s+(.+?)\??$",
        query.strip(),
        re.I,
    )

    if not match:
        return None

    subject = re.sub(
        r"\b(the|a|an)\b",
        "",
        match.group(1),
        flags=re.I,
    )

    cleaned_subject = clean_text(subject).lower()

    return cleaned_subject or None


def get_content_terms(text: str) -> set[str]:
    """
    Extract meaningful words while ignoring stopwords.
    """
    return {
        word
        for word in tokenize(text)
        if word not in STOPWORDS
    }


def calculate_sentence_score(
    query: str,
    sentence: str,
    chunk_score: float,
) -> float:
    """
    Assign a relevance score to a sentence.
    """
    query_terms = get_content_terms(query)
    sentence_terms = get_content_terms(sentence)

    score = (2 * len(query_terms & sentence_terms)) + chunk_score

    subject = extract_definition_subject(query)

    lowered_sentence = f" {sentence.lower()} "

    # Boost sentences that directly define the subject
    if subject and subject in lowered_sentence:
        score += 4

        escaped_subject = re.escape(subject)

        definition_match = re.search(
            rf"\b{escaped_subject}\b[\w\s,()/-]{{0,100}}"
            rf"\b(is|are|refers to|means)\b",
            lowered_sentence,
        )

        if definition_match:
            score += 4

    if lowered_sentence.strip().startswith(
        ("unlike ", "however ", "although ")
    ):
        score -= 1

    return score


def build_extractive_answer(
    query: str,
    chunks: list[dict[str, Any]],
    max_sentences: int = 4,
) -> str:
    """
    Create an answer by selecting the most relevant sentences.
    """
    ranked_sentences: list[tuple[float, int, str]] = []

    for chunk_index, chunk in enumerate(chunks, start=1):

        chunk_score = float(chunk.get("score") or 0.0)

        sentences = split_into_sentences(
            chunk.get("text", "")
        )

        if not sentences and chunk.get("text"):
            sentences = [
                clean_text(chunk["text"])[:500]
            ]

        for sentence in sentences:
            relevance = calculate_sentence_score(
                query,
                sentence,
                chunk_score,
            )

            ranked_sentences.append(
                (relevance, chunk_index, sentence)
            )

    if not ranked_sentences:
        return (
            "I could not find enough relevant information "
            "inside the indexed documents."
        )

    ranked_sentences.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    selected_sentences: list[str] = []
    seen_sentences: set[str] = set()

    for _, chunk_index, sentence in ranked_sentences:

        normalized_sentence = sentence.lower()

        if normalized_sentence in seen_sentences:
            continue

        selected_sentences.append(
            f"{sentence} [{chunk_index}]"
        )

        seen_sentences.add(normalized_sentence)

        if len(selected_sentences) >= max_sentences:
            break

    return (
        "Based on the retrieved documents, "
        + " ".join(selected_sentences)
    )


def generate_answer(
    query: str,
    context: list[str] | list[dict[str, Any]],
) -> str:
    """
    Generate a grounded answer from retrieved context chunks.
    """
    chunks = normalize_context(context)

    if not chunks:
        return (
            "I could not find relevant context "
            "for the given question."
        )

    combined_context = "\n\n".join(
        chunk.get("text", "")
        for chunk in chunks
    )

    # Try transformer generation first
    answer = try_transformer_generation(
        query,
        combined_context,
    )

    # Fallback to extractive summarization
    if answer is None:
        answer = build_extractive_answer(
            query,
            chunks,
        )

    source_references = [
        build_source_label(chunk, idx)
        for idx, chunk in enumerate(chunks, start=1)
    ]

    return (
        f"{answer}\n\nSources:\n"
        + "\n".join(source_references)
    )