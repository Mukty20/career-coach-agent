"""CV/Resume analyzer tool - extracts text from an uploaded PDF."""

from pypdf import PdfReader
import io


def extract_cv_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        full_text = "\n".join(text_parts).strip()

        if not full_text:
            return "ERROR: Could not extract text from this PDF. It may be a scanned image rather than text-based."

        return full_text

    except Exception as e:
        return f"ERROR: Failed to read PDF - {str(e)}"


def summarize_cv_stats(cv_text: str) -> dict:
    if cv_text.startswith("ERROR"):
        return {"word_count": 0, "char_count": 0, "valid": False}

    return {
        "word_count": len(cv_text.split()),
        "char_count": len(cv_text),
        "valid": True,
    }
