"""RapidOCR model wrapper for inference."""

from __future__ import annotations

import logging
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
from rapidocr_pdf import RapidOCRPDF

logger = logging.getLogger(__name__)


class OCRModel:
    """Wrapper for RapidOCR model inference."""

    def __init__(self) -> None:
        self._ocr: RapidOCR | None = None
        self._pdf_ocr: RapidOCRPDF | None = None

    def load(self) -> None:
        """Load the OCR engines."""
        logger.info("Loading RapidOCR engine...")
        self._ocr = RapidOCR()
        self._pdf_ocr = RapidOCRPDF()
        logger.info("RapidOCR loaded successfully")

    @property
    def ocr(self) -> RapidOCR:
        if self._ocr is None:
            self.load()
        return self._ocr  # type: ignore[return-value]

    @property
    def pdf_ocr(self) -> RapidOCRPDF:
        if self._pdf_ocr is None:
            self.load()
        return self._pdf_ocr  # type: ignore[return-value]

    def _image_to_array(self, image: Image.Image | bytes | str | Path) -> np.ndarray:
        """Convert various image formats to numpy array."""
        if isinstance(image, bytes):
            from io import BytesIO

            image = Image.open(BytesIO(image))

        if isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))

        if isinstance(image, (str, Path)):
            return np.array(Image.open(image).convert("RGB"))

        msg = f"Unsupported image type: {type(image)}"
        raise TypeError(msg)

    def predict(self, image: Image.Image | bytes | str | Path) -> str:
        """Run OCR prediction on an image.

        Args:
            image: PIL Image, bytes, file path

        Returns:
            The extracted text from the image
        """
        img_array = self._image_to_array(image)
        result = self.ocr(img_array)

        # result is (detections, timing_info)
        # detections is list of [box, text, confidence]
        detections = result[0]
        if detections is None:
            return ""

        texts = [item[1] for item in detections]
        return "\n".join(texts)

    def predict_with_boxes(self, image: Image.Image | bytes | str | Path) -> list[dict[str, Any]]:
        """Run OCR and return text with bounding boxes.

        Args:
            image: PIL Image, bytes, file path

        Returns:
            List of dicts with 'text', 'confidence', and 'box' keys
        """
        img_array = self._image_to_array(image)
        result = self.ocr(img_array)

        detections = result[0]
        if detections is None:
            return []

        items = []
        for detection in detections:
            box, text, confidence = detection
            items.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "box": box,
                }
            )
        return items

    def process_pdf(
        self,
        pdf_path: str | Path | bytes,
    ) -> list[dict[str, str | int]]:
        """Process a PDF file and extract text from each page.

        Args:
            pdf_path: Path to PDF file or PDF bytes

        Returns:
            List of dicts with page number and extracted text
        """
        if isinstance(pdf_path, bytes):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(pdf_path)
                temp_path = f.name
            result = self.pdf_ocr(temp_path)
            Path(temp_path).unlink()
        else:
            result = self.pdf_ocr(pdf_path)

        if result is None:
            return []

        # rapidocr-pdf returns list of [page_num, text, ???]
        pages = []
        for page_result in result:
            page_num = int(page_result[0]) + 1  # 0-indexed to 1-indexed
            page_text = page_result[1] if len(page_result) > 1 else ""
            pages.append({"page": page_num, "text": page_text})

        return pages


@lru_cache(maxsize=1)
def get_model() -> OCRModel:
    """Get or create the singleton model instance."""
    model = OCRModel()
    model.load()
    return model
