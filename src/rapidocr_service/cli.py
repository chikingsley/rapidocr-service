"""CLI for RapidOCR."""

import sys
from pathlib import Path

from alive_progress import alive_bar

from .model import get_model


def main():
    if len(sys.argv) < 2:
        print("Usage: ocr <pdf_file>", file=sys.stderr)
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    model = get_model()
    total_pages = model.get_pdf_page_count(pdf_path)
    results = []

    with alive_bar(total_pages, title="OCR Processing") as bar:
        for page_result in model.process_pdf_pages(pdf_path):
            results.append(page_result)
            bar()

    # Write to markdown file
    output_path = pdf_path.with_suffix(".md")
    with output_path.open("w") as f:
        f.write(f"# {pdf_path.stem}\n\n")
        for page in results:
            f.write(f"## Page {page['page']}\n\n")
            f.write(page["text"])
            f.write("\n\n")

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
