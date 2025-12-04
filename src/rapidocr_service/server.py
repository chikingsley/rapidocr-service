"""FastAPI server for RapidOCR."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .model import get_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load model on startup."""
    logger.info("Loading RapidOCR...")
    get_model()
    logger.info("RapidOCR loaded and ready")
    yield


app = FastAPI(
    title="RapidOCR Service",
    description="A FastAPI service for OCR using RapidOCR with ONNX Runtime",
    version="0.1.0",
    lifespan=lifespan,
)


class OCRResponse(BaseModel):
    """Response model for OCR endpoints."""

    text: str


class OCRBoxItem(BaseModel):
    """Single OCR result with bounding box."""

    text: str
    confidence: float | None
    box: list[list[float]] | None


class OCRDetailedResponse(BaseModel):
    """Response model for detailed OCR with boxes."""

    items: list[OCRBoxItem]


class PDFPageResult(BaseModel):
    """Result for a single PDF page."""

    page: int
    text: str


class PDFResponse(BaseModel):
    """Response model for PDF OCR endpoints."""

    pages: list[PDFPageResult]
    total_pages: int


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check if the service is healthy and model is loaded."""
    try:
        model = get_model()
        return HealthResponse(status="healthy", model_loaded=model._ocr is not None)
    except Exception:
        return HealthResponse(status="unhealthy", model_loaded=False)


@app.post("/ocr", response_model=OCRResponse)
async def ocr(
    file: Annotated[UploadFile, File(description="Image file to process")],
) -> OCRResponse:
    """Run OCR on an uploaded image and return extracted text."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected an image file.",
        )

    try:
        image_bytes = await file.read()
        model = get_model()
        result = model.predict(image_bytes)
        return OCRResponse(text=result)
    except Exception as e:
        logger.exception("Error processing image")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ocr/detailed", response_model=OCRDetailedResponse)
async def ocr_detailed(
    file: Annotated[UploadFile, File(description="Image file to process")],
) -> OCRDetailedResponse:
    """Run OCR and return text with bounding boxes and confidence scores."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected an image file.",
        )

    try:
        image_bytes = await file.read()
        model = get_model()
        results = model.predict_with_boxes(image_bytes)
        items = [OCRBoxItem(**r) for r in results]  # type: ignore[arg-type]
        return OCRDetailedResponse(items=items)
    except Exception as e:
        logger.exception("Error processing image")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ocr/pdf", response_model=PDFResponse)
async def ocr_pdf(
    file: Annotated[UploadFile, File(description="PDF file to process")],
) -> PDFResponse:
    """Process a PDF file and extract text from each page."""
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected a PDF file.",
        )

    try:
        pdf_bytes = await file.read()
        model = get_model()
        results = model.process_pdf(pdf_bytes)
        pages = [PDFPageResult(page=r["page"], text=r["text"]) for r in results]  # type: ignore[arg-type]
        return PDFResponse(pages=pages, total_pages=len(pages))
    except Exception as e:
        logger.exception("Error processing PDF")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.exception_handler(Exception)
async def global_exception_handler(_request, _exc):
    """Handle uncaught exceptions."""
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
