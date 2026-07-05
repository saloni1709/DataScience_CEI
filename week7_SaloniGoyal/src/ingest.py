from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SUPPORTED_FILE_TYPES = {
    ".txt",
    ".md",
    ".pdf",
    ".json",
}


IGNORED_METADATA_FILES = {
    "answers.json",
    "pdf_urls.json",
    "qrels.json",
    "queries.json",
}


NOISY_SECTION_HEADERS = (
    "# references",
    "## references",
    "references",
    "# supplementary material",
    "## supplementary material",
    "supplementary material",
)


def clean_text(text: str) -> str:
    """
    Normalize whitespace while preserving meaning.
    """
    return re.sub(r"\s+", " ", text).strip()


def is_supported_document(path: Path) -> bool:
    """
    Check whether a file should be processed.
    """
    if path.suffix.lower() not in SUPPORTED_FILE_TYPES:
        return False

    if path.name in IGNORED_METADATA_FILES:
        return False

   
    if any(part.startswith(".") for part in path.parts):
        return False

    
    if "__pycache__" in path.parts:
        return False

    return True


def collect_supported_files(
    input_path: Path,
    max_files: int | None = None,
) -> list[Path]:
    """
    Gather all supported files from a directory.
    """
    if not input_path.exists():
        return []

    if input_path.is_file():
        files = (
            [input_path]
            if is_supported_document(input_path)
            else []
        )

    else:
        files = [
            file_path
            for file_path in sorted(input_path.rglob("*"))
            if file_path.is_file()
            and is_supported_document(file_path)
        ]

    return (
        files[:max_files]
        if max_files is not None
        else files
    )


def read_text_document(path: Path) -> list[dict[str, Any]]:
    """
    Load plain text or markdown documents.
    """
    text = clean_text(
        path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    )

    if not text:
        return []

    return [
        {
            "text": text,
            "source": path.name,
            "path": str(path),
            "title": path.stem,
        }
    ]


def read_pdf_document(path: Path) -> list[dict[str, Any]]:
    """
    Extract text page-by-page from PDF files.
    """
    try:
        from pypdf import PdfReader

    except ImportError as error:
        raise RuntimeError(
            "PDF support requires pypdf.\n"
            "Install it using: pip install -r requirements.txt"
        ) from error

    extracted_documents: list[dict[str, Any]] = []

    reader = PdfReader(str(path))

    for page_number, page in enumerate(
        reader.pages,
        start=1,
    ):
        text = clean_text(page.extract_text() or "")

        if not text:
            continue

        extracted_documents.append(
            {
                "text": text,
                "source": path.name,
                "path": str(path),
                "title": path.stem,
                "page": page_number,
            }
        )

    return extracted_documents


def is_noisy_section(text: str) -> bool:
    """
    Detect sections that are usually irrelevant for retrieval.
    """
    lowered = text.lstrip().lower()

    return lowered.startswith(
        NOISY_SECTION_HEADERS
    )


def extract_json_records(
    payload: Any,
    path: Path,
) -> list[dict[str, Any]]:
    """
    Convert JSON content into normalized text records.
    """
    records: list[dict[str, Any]] = []

    if (
        isinstance(payload, dict)
        and isinstance(payload.get("sections"), list)
    ):
        title = str(payload.get("title") or path.stem)

        document_id = (
            payload.get("id")
            or path.stem
        )

        authors = payload.get("authors") or []
        categories = payload.get("categories") or []

        for section in payload["sections"]:

            if not isinstance(section, dict):
                continue

            text = clean_text(
                str(section.get("text") or "")
            )

            if not text or is_noisy_section(text):
                continue

            records.append(
                {
                    "text": text,
                    "source": path.name,
                    "path": str(path),
                    "title": title,
                    "document_id": document_id,
                    "authors": authors,
                    "categories": categories,
                    "section_id": section.get(
                        "section_id"
                    ),
                }
            )

        return records

    
    if isinstance(payload, dict):

        for field in (
            "text",
            "content",
            "body",
            "abstract",
        ):
            text = clean_text(
                str(payload.get(field) or "")
            )

            if not text:
                continue

            records.append(
                {
                    "text": text,
                    "source": path.name,
                    "path": str(path),
                    "title": str(
                        payload.get("title")
                        or path.stem
                    ),
                    "document_id": (
                        payload.get("id")
                        or path.stem
                    ),
                }
            )

            break

        return records

    
    if isinstance(payload, list):

        for index, item in enumerate(payload):

            if not isinstance(item, dict):
                continue

            for field in (
                "text",
                "content",
                "body",
                "abstract",
            ):
                text = clean_text(
                    str(item.get(field) or "")
                )

                if not text:
                    continue

                records.append(
                    {
                        "text": text,
                        "source": path.name,
                        "path": str(path),
                        "title": str(
                            item.get("title")
                            or path.stem
                        ),
                        "document_id": (
                            item.get("id")
                            or f"{path.stem}-{index}"
                        ),
                    }
                )

                break

    return records


def read_json_document(
    path: Path,
) -> list[dict[str, Any]]:
    """
    Read and process JSON files.
    """
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    return extract_json_records(
        payload,
        path,
    )


def load_documents(
    input_path: str | Path,
    max_files: int | None = None,
) -> list[dict[str, Any]]:
    """
    Load all supported documents and convert them
    into normalized text records.
    """
    path = Path(input_path)

    loaded_documents: list[dict[str, Any]] = []

    for file_path in collect_supported_files(
        path,
        max_files=max_files,
    ):
        suffix = file_path.suffix.lower()

        if suffix in {".txt", ".md"}:
            loaded_documents.extend(
                read_text_document(file_path)
            )

        elif suffix == ".pdf":
            loaded_documents.extend(
                read_pdf_document(file_path)
            )

        elif suffix == ".json":
            loaded_documents.extend(
                read_json_document(file_path)
            )

    return loaded_documents


def count_supported_files(
    input_path: str | Path,
) -> int:
    """
    Count indexable files without loading content.
    """
    return len(
        collect_supported_files(
            Path(input_path)
        )
    )


def chunk_text(
    text: str,
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[str]:
    """
    Split text into overlapping chunks.
    """
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than zero"
        )

    if overlap < 0:
        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    words = text.split()

    if not words:
        return []

    chunks: list[str] = []

    current_index = 0

    step_size = chunk_size - overlap

    while current_index < len(words):

        end_index = current_index + chunk_size

        chunk = " ".join(
            words[current_index:end_index]
        )

        chunks.append(chunk)

        current_index += step_size

    return chunks


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 180,
    overlap: int = 40,
) -> list[dict[str, Any]]:
    """
    Convert full documents into retrieval-ready chunks.
    """
    all_chunks: list[dict[str, Any]] = []

    for document_index, document in enumerate(
        documents
    ):
        text_chunks = chunk_text(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_index, chunk_text_value in enumerate(
            text_chunks
        ):
            metadata = {
                key: value
                for key, value in document.items()
                if key != "text"
            }

            all_chunks.append(
                {
                    **metadata,
                    "id": f"chunk-{len(all_chunks)}",
                    "document_index": document_index,
                    "chunk_index": chunk_index,
                    "text": chunk_text_value,
                }
            )

    return all_chunks


def ingest_documents(
    input_path: str | Path,
    chunk_size: int = 180,
    overlap: int = 40,
    max_files: int | None = None,
) -> list[dict[str, Any]]:
    """
    Full ingestion pipeline:
    load documents -> split into chunks.
    """
    documents = load_documents(
        input_path,
        max_files=max_files,
    )

    return chunk_documents(
        documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )