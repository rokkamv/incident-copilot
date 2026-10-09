# 0001: Python, FastAPI and pgvector

**Status:** accepted

## Decision

Build the copilot in Python with FastAPI, use the Claude API through the official Anthropic SDK,
and store embeddings in PostgreSQL with pgvector.

## Why

- Most AI libraries ship for Python first, and FastAPI is close to Flask and Django.
- pgvector keeps vectors in a database we already know how to run, back up and query with SQL.

## Rejected

- A dedicated vector database: one more service to run, with no benefit at this data size.
- Rails or Spring Boot: workable, but AI tooling and examples lag behind Python.
