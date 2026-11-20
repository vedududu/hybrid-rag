import re
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader


CODE_ROOT = Path("../fastapi-src")
INCLUDE_PATTERN = re.compile(r"\{\*\s*(\S+)\s*(.*?)\*\}|\{!\s*(\S+?)\s*!\}")
LANGUAGES = {".py": "python", ".html": "html"}


def pick_lines(lines, spec):
    """Keep only the lines named in a spec like '6,19:20' (counting from 1, both ends included)."""
    picked = []
    for part in spec.split(","):
        start, _, end = part.partition(":")
        picked.extend(lines[int(start) - 1:int(end or start)])
    return picked


def expand_includes(text):
    """Replace FastAPI's code placeholders with the actual code they point to."""
    def replace(match):
        relative = match.group(1) or match.group(3)
        options = match.group(2) or ""
        path = CODE_ROOT / relative.removeprefix("../../")
        if not path.exists():
            return match.group(0)
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        line_range = re.search(r"ln\[([^\]]+)\]", options)
        if line_range:
            lines = pick_lines(lines, line_range.group(1))
        language = LANGUAGES.get(path.suffix, "")
        return f"```{language}\n" + "\n".join(lines) + "\n```"
    return INCLUDE_PATTERN.sub(replace, text)



SUPPORTED = {".md", ".txt", ".html", ".pdf"}


def find_files(folder):
    """Return every supported document inside a folder (including subfolders)."""
    files = []
    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            files.append(path)
    return files


def read_text_file(path):
    """Read a .md or .txt file and return its text."""
    return path.read_text(encoding="utf-8-sig")


def read_html_file(path):
    """Read an .html file and return just its visible text."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8-sig"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def read_pdf_file(path):
    """Read a .pdf file and return the text of all its pages."""
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_document(path):
    """Read a file and return its text together with where it came from."""
    suffix = path.suffix.lower()
    if suffix == ".html":
        text = read_html_file(path)
    elif suffix == ".pdf":
        text = read_pdf_file(path)
    else:
        text = read_text_file(path)
    if suffix == ".md":
        text = expand_includes(text)
    return {
        "text": text,
        "source": path.name,
        "path": str(path),
        "type": suffix,
    }


def load_all(folder):
    """Load every supported document in a folder."""
    documents = []
    for path in find_files(folder):
        documents.append(load_document(path))
    return documents
