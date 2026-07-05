# Document Question Answering System using RAG

This repository shows a simple way to build a document-based question answering system with Retrieval-Augmented Generation. It looks through academic papers, picks out the most relevant passages, and uses them as supporting context for answering a query.

The design stays lightweight and works offline by relying on a local TF-IDF index rather than external APIs.

## What this project does

The workflow in this project can:

- read documents from the folder data/open_ragbench/pdf/arxiv/corpus
- break long texts into smaller, overlapping chunks
- save those chunks in a local search index
- match a question to the most relevant pieces of text
- produce an answer grounded in the retrieved context

## Why this project was built

This project was created as a hands-on example for understanding how retrieval and generation can be combined in a practical setting. It is meant to make the core idea of RAG easier to see and experiment with.

## Project workflow

1. Load documents from the dataset directory.
2. Split each document into smaller overlapping segments.
3. Build a local TF-IDF search index.
4. Find the chunks that best match the user question.
5. Generate a final answer using those retrieved passages.

## Project structure

```text
Document Question Answering System (RAG)/
|-- data/
|   `-- open_ragbench/
|       `-- pdf/
|           `-- arxiv/
|               `-- corpus/
|-- src/
|   |-- ingest.py
|   |-- retrieve.py
|   |-- generate.py
|   `-- app.py
|-- vector_store/
|-- requirements.txt
`-- README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The implementation uses standard Python tools for reading text and JSON files, while pypdf is the only extra package needed for PDF handling.

## Data folder

The default dataset location used by the app is:

```text
data/open_ragbench/pdf/arxiv/corpus
```

The application will use this folder automatically when it is present.

## Run the project

### 1. Build the index and ask one question

```bash
python src/app.py --rebuild --query "What is the main idea of the document?"
```

### 2. Ask another question using the saved index

```bash
python src/app.py --query "What are the key findings?"
```

### 3. Run the app interactively

```bash
python src/app.py --interactive
```

### 4. See example questions from the dataset

```bash
python src/app.py --sample-queries 5
```

## Sample input and output

Example input:

```text
What is the main idea of the document?
```

Example output:

```text
Based on the retrieved documents, ...

Sources:
[1] document_name - source_file
```

## How the code is organized

- ingest.py: loads documents and divides them into smaller chunks
- retrieve.py: builds the local TF-IDF search index
- generate.py: turns retrieved context into a readable answer
- app.py: connects the workflow and runs the CLI

## Useful options

```bash
python src/app.py --help
```

A few helpful flags are:

- --rebuild: rebuild the index
- --top-k 5: retrieve more chunks
- --chunk-size 220: use larger chunks
- --overlap 50: add more overlap between chunks
- --max-files 0: index all available files
- --documents path/to/folder: use your own folder or file

## Note for viva or interview

A short way to describe the project is:

- first find the most relevant information
- then use that information to answer the question

That is the core idea behind RAG, and this project shows it in a simple, practical form.
