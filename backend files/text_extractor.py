import os
import re


def extract_text(file_path: str) -> str:
    """
    Extract plain text from PDF, DOCX, or TXT files.
    Returns the extracted text string or raises ValueError on failure.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return _extract_txt(file_path)
    elif ext == ".pdf":
        return _extract_pdf(file_path)
    elif ext in (".doc", ".docx"):
        return _extract_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ─────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────

def _extract_txt(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def _extract_pdf(file_path: str) -> str:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages).strip()
    except ImportError:
        # fallback to PyPDF2
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            texts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    texts.append(t)
            return "\n\n".join(texts).strip()
        except ImportError:
            raise ImportError(
                "Install pdfplumber or PyPDF2: pip install pdfplumber"
            )


def _extract_docx(file_path: str) -> str:
    try:
        from docx import Document
        doc = Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs).strip()
    except ImportError:
        raise ImportError(
            "Install python-docx: pip install python-docx"
        )


def clean_text(text: str) -> str:
    """Remove excessive whitespace and normalize line endings."""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()
