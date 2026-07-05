from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .generate import generate_answer
    from .ingest import ingest_documents, count_supported_files
    from .retrieve import (
        RAG_INDEX_FILENAME,
        TfidfVectorStore,
        build_vector_store,
        load_vector_store,
    )
except ImportError:
    from generate import generate_answer
    from ingest import ingest_documents, count_supported_files
    from retrieve import (
        RAG_INDEX_FILENAME,
        TfidfVectorStore,
        build_vector_store,
        load_vector_store,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
VECTOR_STORE_DIR = PROJECT_ROOT / "vector_store"

OPEN_RAGBENCH_ROOT = DATA_DIR / "open_ragbench"
OPEN_RAGBENCH_CORPUS = OPEN_RAGBENCH_ROOT / "pdf" / "arxiv" / "corpus"
OPEN_RAGBENCH_QUERIES = OPEN_RAGBENCH_ROOT / "pdf" / "arxiv" / "queries.json"

LEGACY_CORPUS = (
    PROJECT_ROOT / "dataset" / "open_ragbench" / "pdf" / "arxiv" / "corpus"
)


def resolve_path(path_value: str | Path) -> Path:
    """
    Convert relative paths into absolute project paths.
    """
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_source_directory(args: argparse.Namespace) -> Path:
    """
    Decide which document source should be indexed.
    Priority:
    1. User-provided path
    2. Legacy dataset
    3. Default Open RAGBench dataset
    4. Generic data folder
    """
    if args.documents:
        return resolve_path(args.documents)

    if args.use_dataset and LEGACY_CORPUS.exists():
        return LEGACY_CORPUS

    if OPEN_RAGBENCH_CORPUS.exists():
        return OPEN_RAGBENCH_CORPUS

    return DATA_DIR


def get_file_limit(args: argparse.Namespace) -> int | None:
    """
    Returns None when all files should be indexed.
    """
    return None if args.max_files <= 0 else args.max_files


def create_metadata(args: argparse.Namespace, source_dir: Path) -> dict:
    """
    Store configuration details used during indexing.
    This helps detect when the index becomes outdated.
    """
    return {
        "source_path": str(source_dir.resolve()),
        "use_dataset": args.use_dataset,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "max_files": get_file_limit(args) or 0,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple local RAG application for querying documents."
    )

    parser.add_argument(
        "--documents",
        help=(
            "Path to a document or folder to index. "
            "Defaults to the Open RAGBench corpus if available."
        ),
    )

    parser.add_argument(
        "--vector-store",
        default=str(VECTOR_STORE_DIR),
        help="Directory where vector index files are stored.",
    )

    parser.add_argument(
        "--query",
        help="Question to ask the RAG system.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of relevant chunks to retrieve.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=180,
        help="Size of each text chunk in words.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=40,
        help="Word overlap between chunks.",
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=25,
        help="Maximum number of files to process. Use 0 for all files.",
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuilding the vector database.",
    )

    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Use the older bundled Open RAGBench dataset if available.",
    )

    parser.add_argument(
        "--sample-queries",
        type=int,
        default=0,
        help="Display sample dataset questions and exit.",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep asking questions until manually exited.",
    )

    return parser.parse_args()


def build_store(args: argparse.Namespace) -> TfidfVectorStore:
    """
    Read documents, create chunks,
    and build a TF-IDF vector store.
    """
    source_dir = get_source_directory(args)

    chunks = ingest_documents(
        source_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_files=get_file_limit(args),
    )

    if not chunks:
        raise SystemExit(
            "No documents were found for indexing.\n"
            "Add files to data/open_ragbench/pdf/arxiv/corpus "
            "or use --documents to specify a custom location."
        )

    store_path = resolve_path(args.vector_store)

    vector_store = build_vector_store(
        chunks,
        output_dir=store_path,
        metadata=create_metadata(args, source_dir),
    )

    print(f"Indexed {vector_store.size} chunks from:\n{source_dir}")

    return vector_store


def load_or_create_store(args: argparse.Namespace) -> TfidfVectorStore:
    """
    Load an existing vector store if possible.
    Otherwise rebuild it automatically.
    """
    store_path = resolve_path(args.vector_store)
    index_file = store_path / RAG_INDEX_FILENAME

    if args.rebuild or not index_file.exists():
        return build_store(args)

    store = load_vector_store(store_path)

    expected_metadata = create_metadata(
        args,
        get_source_directory(args),
    )

    if store.metadata != expected_metadata:
        print(
            "Existing vector store settings do not match current configuration."
        )
        print("Rebuilding vector store...\n")

        return build_store(args)

    print(f"Loaded {store.size} indexed chunks.")

    return store


def answer_query(
    store: TfidfVectorStore,
    query: str,
    top_k: int,
) -> None:
    """
    Retrieve relevant chunks and generate a response.
    """
    retrieved_chunks = store.search(query, top_k=top_k)

    response = generate_answer(query, retrieved_chunks)

    print("\n")
    print(response)
    print("\n")


def show_sample_queries(limit: int) -> None:
    """
    Print a few sample questions from the dataset.
    """
    if not OPEN_RAGBENCH_QUERIES.exists():
        raise SystemExit(
            "queries.json could not be found in the Open RAGBench dataset."
        )

    with OPEN_RAGBENCH_QUERIES.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    for idx, item in enumerate(queries.values(), start=1):
        if idx > limit:
            break

        question = item.get("query") if isinstance(item, dict) else None

        if question:
            print(f"{idx}. {question}")


def main() -> None:
    args = parse_arguments()

    if args.sample_queries > 0:
        show_sample_queries(args.sample_queries)
        return

    source_dir = get_source_directory(args)

    if args.rebuild:
        total_files = count_supported_files(source_dir)

        chosen_files = (
            total_files
            if get_file_limit(args) is None
            else min(total_files, args.max_files)
        )

        print(f"Source directory : {source_dir}")
        print(f"Files detected   : {total_files}")
        print(f"Files selected   : {chosen_files}\n")

    store = load_or_create_store(args)

    if args.query:
        answer_query(store, args.query, args.top_k)
        return

    while True:
        try:
            user_query = input(
                "Ask something (press Enter to quit): "
            ).strip()

        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not user_query:
            return

        answer_query(store, user_query, args.top_k)

        if not args.interactive:
            return


if __name__ == "__main__":
    main()