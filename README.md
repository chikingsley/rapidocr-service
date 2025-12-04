# RapidOCR Service

A FastAPI service for OCR using [RapidOCR](https://github.com/RapidAI/RapidOCR) with ONNX Runtime.

## Features

- Fast OCR text extraction
- Bounding box and confidence scores
- PDF processing (multi-page)
- ONNX Runtime backend (CPU optimized)

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker (for containerized deployment)

## Quick Start

### Local Development

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the server
uv run uvicorn rapidocr_service.server:app --reload
```

### Docker

```bash
docker compose up --build
```

## API Endpoints

### Health Check

```bash
curl http://localhost:8000/health
```

### OCR (Image)

```bash
curl -X POST http://localhost:8000/ocr \
  -F "file=@image.png"
```

### OCR with Bounding Boxes

```bash
curl -X POST http://localhost:8000/ocr/detailed \
  -F "file=@image.png"
```

### OCR PDF

```bash
curl -X POST http://localhost:8000/ocr/pdf \
  -F "file=@document.pdf"
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run linter
uv run ruff check .

# Run formatter
uv run ruff format .

# Run type checker
uv run ty check

# Run tests
uv run pytest
```

## License

MIT
