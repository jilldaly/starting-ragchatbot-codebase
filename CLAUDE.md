# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Always use `uv` to manage dependencies and run Python — never use `pip` directly.

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package>

# Run a Python file
uv run script.py

# Run the application
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

The app is available at `http://localhost:8000`. API docs at `http://localhost:8000/docs`.

Requires a `.env` file in the project root with `ANTHROPIC_API_KEY=...` (see `.env.example`).

## Architecture

Full-stack RAG chatbot: a vanilla JS frontend served as static files by a FastAPI backend. The backend uses ChromaDB for vector storage and Claude (via Anthropic SDK) for generation.

### Query flow

1. Frontend POSTs `{ query, session_id }` to `POST /api/query`
2. `RAGSystem.query()` wraps the prompt and fetches conversation history from `SessionManager`
3. `AIGenerator.generate_response()` calls Claude with the `search_course_content` tool available
4. If Claude invokes the tool, `CourseSearchTool` runs a semantic search via `VectorStore` and returns formatted chunks; Claude makes a second call to synthesize the final answer
5. Sources and answer are returned to the frontend

### Tool-use pattern

`AIGenerator` handles a two-call Claude loop: the first call may return `stop_reason: "tool_use"`, which triggers `_handle_tool_execution()` — it runs the tool, appends results to the message history, then makes a second call (without tools) for the final answer. Only one tool exists: `search_course_content` (defined in `search_tools.py`).

### Vector store

Two ChromaDB collections (in `./chroma_db/`):
- `course_catalog` — one document per course (title, instructor, lesson links); used for fuzzy course name resolution
- `course_content` — sentence-based chunks with `course_title` and `lesson_number` metadata; used for semantic search with optional filtering

Course title doubles as the unique ID in `course_catalog`.

### Document ingestion

On startup, `app.py` calls `RAGSystem.add_course_folder("../docs")`. `DocumentProcessor` parses `.txt`/`.pdf`/`.docx` files expecting this header format:
```
Course Title: ...
Course Link: ...
Course Instructor: ...
Lesson 1: Title
Lesson Link: ...
<content>
```
Text is split into sentence-based chunks (`CHUNK_SIZE=800`, `CHUNK_OVERLAP=100`). Already-indexed courses (matched by title) are skipped.

### Session management

`SessionManager` keeps an in-memory dict of sessions. History is stored as flat `User:/Assistant:` strings and injected into the Claude system prompt. Rolling window capped at `MAX_HISTORY * 2` (default: 4) messages.

### Configuration

All tuneable values live in `backend/config.py`: model name, chunk size/overlap, max search results, max history, ChromaDB path.
