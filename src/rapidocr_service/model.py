"""RapidOCR model wrapper for inference."""

from __future__ import annotations

import logging
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

import fitz
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)


class OCRModel:
    """Wrapper for RapidOCR model inference."""

    def __init__(self) -> None:
        self._ocr: RapidOCR | None = None

    def load(self) -> None:
        """Load the OCR engine."""
        logger.info("Loading RapidOCR engine...")
        self._ocr = RapidOCR()
        logger.info("RapidOCR loaded successfully")

    @property
    def ocr(self) -> RapidOCR:
        if self._ocr is None:
            self.load()
        return self._ocr  # type: ignore[return-value]

    def _image_to_array(self, image: Image.Image | bytes | str | Path) -> np.ndarray:
        """Convert various image formats to numpy array."""
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        if isinstance(image, (str, Path)):
            return np.array(Image.open(image).convert("RGB"))

        msg = f"Unsupported image type: {type(image)}"
        raise TypeError(msg)

    def predict(self, image: Image.Image | bytes | str | Path) -> str:
        """Run OCR prediction on an image."""
        img_array = self._image_to_array(image)
        result = self.ocr(img_array)

        detections = result[0]
        if detections is None:
            return ""

        texts = [item[1] for item in detections]
        return "\n".join(texts)

    def predict_with_boxes(self, image: Image.Image | bytes | str | Path) -> list[dict[str, Any]]:
        """Run OCR and return text with bounding boxes."""
        img_array = self._image_to_array(image)
        result = self.ocr(img_array)

        detections = result[0]
        if detections is None:
            return []

        items = []
        for detection in detections:
            box, text, confidence = detection
            items.append({
                "text": text,
                "confidence": float(confidence),
                "box": box,
            })
        return items

    def get_pdf_page_count(self, pdf_path: str | Path | bytes) -> int:
        """Get the number of pages in a PDF."""
        if isinstance(pdf_path, bytes):
            doc = fitz.open(stream=pdf_path, filetype="pdf")
        else:
            doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count

    def process_pdf_pages(
        self,
        pdf_path: str | Path | bytes,
        dpi: int = 150,
    ) -> Generator[dict[str, str | int], None, None]:
        """Process PDF page by page, yielding results as they complete."""
        if isinstance(pdf_path, bytes):
            doc = fitz.open(stream=pdf_path, filetype="pdf")
        else:
            doc = fitz.open(pdf_path)

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Render page to image
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                # Run OCR
                text = self.predict(img)
                yield {"page": page_num + 1, "text": text}
        finally:
            doc.close()

    def process_pdf(
        self,
        pdf_path: str | Path | bytes,
        dpi: int = 150,
    ) -> list[dict[str, str | int]]:
        """Process a PDF file and extract text from each page."""
        return list(self.process_pdf_pages(pdf_path, dpi))


@lru_cache(maxsize=1)
def get_model() -> OCRModel:
    """Get or create the singleton model instance."""
    model = OCRModel()
    model.load()
    return model
