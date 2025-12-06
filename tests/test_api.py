"""Tests for the FastAPI server."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    with patch("rapidocr_service.server.get_model") as mock:
        model = MagicMock()
        model._ocr = True
        model.predict.return_value = "Extracted text from image"
        model.predict_with_boxes.return_value = [
            {"text": "Hello", "confidence": 0.95, "box": [[0, 0], [100, 0], [100, 20], [0, 20]]}
        ]
        model.process_pdf.return_value = [{"page": 1, "text": "Page 1 content"}]
        mock.return_value = model
        yield model


@pytest.fixture
def client(mock_model):
    """Create a test client."""
    from rapidocr_service.server import app

    return TestClient(app)


@pytest.fixture
def sample_image() -> bytes:
    """Create a sample test image."""
    img = Image.new("RGB", (100, 100), color="white")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.read()


def test_health_check(client):
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_ocr_endpoint(client, sample_image, mock_model):
    """Test the OCR endpoint."""
    response = client.post(
        "/ocr",
        files={"file": ("test.png", sample_image, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Extracted text from image"
    mock_model.predict.assert_called_once()


def test_ocr_detailed_endpoint(client, sample_image, mock_model):
    """Test the detailed OCR endpoint."""
    response = client.post(
        "/ocr/detailed",
        files={"file": ("test.png", sample_image, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 1
    assert data["items"][0]["text"] == "Hello"
    assert data["items"][0]["confidence"] == 0.95
    mock_model.predict_with_boxes.assert_called_once()


def test_ocr_pdf_endpoint(client, mock_model):
    """Test the PDF OCR endpoint."""
    # Create a minimal PDF-like bytes (actual PDF would be needed for real test)
    pdf_bytes = b"%PDF-1.4 fake pdf content"

    response = client.post(
        "/ocr/pdf",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_pages"] == 1
    assert data["pages"][0]["page"] == 1
    assert data["pages"][0]["text"] == "Page 1 content"
    mock_model.process_pdf.assert_called_once()


def test_invalid_file_type(client):
    """Test that non-image files are rejected for OCR."""
    response = client.post(
        "/ocr",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_invalid_pdf_type(client):
    """Test that non-PDF files are rejected for PDF OCR."""
    response = client.post(
        "/ocr/pdf",
        files={"file": ("test.png", b"not a pdf", "image/png")},
    )
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
