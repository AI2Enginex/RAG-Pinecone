from PyPDF2 import PdfReader

# ========================== FILE UTILITIES ============================

class ReadFile:
    """
    Utility to read text from PDFs.
    """

    @classmethod
    def read_pdf_pages(cls, file_path: str):
        """
        Reads PDF page-wise and preserves page numbers.
        """
        try:
            reader = PdfReader(file_path)
            pages = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    pages.append({
                        "page_number": i + 1,
                        "text": text
                    })

            print(f"Loaded {len(pages)} pages from {file_path}.")
            return pages

        except Exception as e:
            print(f"Error reading PDF file '{file_path}': {e}")
            return []
