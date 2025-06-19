import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import ssl
from typing import Self

import requests
from requests.adapters import HTTPAdapter
from pypdf import PdfReader, PdfWriter


# TODO: Inspired by https://github.com/metachris/pdfx/blob/master/pdfx/extractor.py, extract references from a paper
# TODO: symlinks
# TODO: notes
# TODO: web interface

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")

    download = subparsers.add_parser("download", help="Download papers from the index file.")
    search = subparsers.add_parser("search", help="Search for papers in the index file.")
    list = subparsers.add_parser("list", help="List all papers in the index file.")

    download.add_argument("-i", "--index", type=Path, default="papiry.md")
    download.add_argument("-o", "--out_dir", type=Path, default="pdf")

    search.add_argument("query", type=str)
    search.add_argument("-i", "--index", type=Path, default="papiry.md")

    list.add_argument("-i", "--index", type=Path, default="papiry.md")

    return parser

class Logger:
    """Singleton logger class for consistent logging throughout the application."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
        return cls._instance
    
    def info(self, message: str):
        print(message)
    
    def warn(self, message: str):
        print(f"\033[33m{message}\033[0m")
    
    def error(self, message: str):
        print(f"\033[31m{message}\033[0m")


# Create global logger instance
logger = Logger()

@dataclass
class Section:
    name: str
    level: int

@dataclass
class Paper:
    filename: str
    sections: list[Section]
    links: list[str]
    line: str

    def get_title(self) -> str:
        result = self.line.replace(f"[{self.filename}]", "").lstrip("-")
        for link in self.links:
            result = result.replace(f": {link}", "")
            result = result.replace(link, "")
        result = result.replace("+", "")
        return result.strip()

    def get_path(self) -> Path:
        return Path(*[section.name for section in self.sections] + [self.filename])

@dataclass
class Index:
    papers: list[Paper]


    @classmethod
    def read_index(cls, index_file: Path) -> Self:
        papers = []
        with open(index_file, "r") as f:
            sections = []
            for i, line in enumerate(f):
                if line.startswith("%"):
                    continue
                line = line.strip()

                if line.startswith("#"):
                    level = len(line) - len(line.lstrip("#"))
                    name = find_inside_brackets(line[level:])
                    if name is None:
                        continue

                    while sections and sections[-1].level >= level:
                        sections.pop()
                    sections.append(Section(name, level))
                elif line.startswith("-"):
                    filename = find_inside_brackets(line[1:])
                    if filename is None:
                        continue
                    urls = find_urls(line[1:])
                    papers.append(Paper(filename, sections.copy(), urls, line))
        return cls(papers)


def find_urls(s: str) -> list[str]:
    return re.findall(r'(https?://\S+)', s)


def find_inside_brackets(s: str) -> str | None:
    match = re.search(r'\[(.+)]', s)
    if match:
        return match.group(1)
    return None


def read_existing(output_dir: Path) -> dict[str, Path]:
    existing = {}
    for f in output_dir.glob("**/*.pdf"):
        name = f.name[:-len(".pdf")]
        if name in existing:
            raise f"Duplicate filename {name} ({f} and {existing[name]})"
        existing[name] = f
    return existing


def download_index(index: Index, existing: dict[str, Path], output_dir: Path):
    for paper in index.papers:
        output_path = (
            output_dir /
            Path(*[section.name for section in paper.sections]) /
            (paper.filename + ".pdf" if not paper.filename.endswith(".pdf") else paper.filename)
        )

        if output_path.exists():
            continue
        if paper.filename in existing:
            logger.info(f"Moving {existing[paper.filename]} to {output_path}...")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            existing[paper.filename].rename(output_path)
            continue
        if len(paper.links) == 0:
            logger.warn(f"No URL found for paper {paper.get_title()} ({paper.get_path()})")
            continue
        pdf_urls = [get_pdf_url(url) for url in paper.links]
        pdf_urls = [u for u in pdf_urls if u is not None]
        if len(pdf_urls) == 0:
            logger.warn(f"Cannot resolve any download URL for {paper.filename} from paper URLs: {", ".join(paper.links)}")
            continue
        download_and_merge_pdfs(pdf_urls, output_path)


def get_pdf_url(paper_url: str) -> str | None:
    if (paper_url.endswith(".pdf")
            or paper_url.startswith("https://openreview.net/pdf")
            or paper_url.startswith("https://openreview.net/attachment")
            or paper_url.startswith("https://dl.acm.org/doi/pdf")
    ):
        return paper_url
    if paper_url.startswith("https://arxiv.org/abs/") or paper_url.startswith("https://www.arxiv.org/abs/"):
        return paper_url.replace("/abs/", "/pdf/")
    if paper_url.startswith("https://arxiv.org/pdf/") or paper_url.startswith("https://www.arxiv.org/pdf/"):
        return paper_url
    if paper_url.startswith("https://nature.com/articles/") or paper_url.startswith("https://www.nature.com/articles/"):
        return paper_url + ".pdf"
    if paper_url.startswith("https://openreview.net/forum?id=") or paper_url.startswith("https://www.openreview.net/forum?id="):
        return paper_url.replace("/forum", "/pdf")
    return None


def download_and_merge_pdfs(pdf_urls: list[str], output_path: Path):
    assert len(pdf_urls) > 0
    if len(pdf_urls) == 1:
        download_pdf(pdf_urls[0], output_path)
        return

    parts_paths = []
    for i, pdf_url in enumerate(pdf_urls):
        part_path = Path(str(output_path)[:-len(".pdf")] + f"-download_part-{i + 1}.pdf")
        download_pdf(pdf_url, part_path)
        parts_paths.append(part_path)

    logger.info(f"Merging {", ".join([p.name for p in parts_paths])} -> {output_path.name}")
    merge_pdfs(output_path, parts_paths)

    for part_path in parts_paths:
        part_path.unlink()


def merge_pdfs(output_path, pdf_paths):
    writer = PdfWriter()

    for pdf_path in pdf_paths:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            writer.add_page(page)

    with open(output_path, "wb") as output_file:
        writer.write(output_file)


def download_pdf(pdf_url: str, output_path: Path):
    logger.info(f"Downloading {pdf_url} to {output_path}...")
    
    # Create SSL context that allows legacy renegotiation.
    ssl_context = ssl.create_default_context()
    ssl_context.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
    
    session = requests.Session()
    
    class SSLAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs["ssl_context"] = ssl_context
            return super().init_poolmanager(*args, **kwargs)
    
    session.mount("https://", SSLAdapter())
    
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/58.0.3029.110 Safari/537.3"
        ),
        "Accept": "application/pdf",
    }

    response = session.get(pdf_url, headers=headers, stream=True, timeout=30)
    if response.status_code == 200:
        content_type = response.headers.get("Content-Type", "")
        if "application/pdf" in content_type or "application/octet-stream" in content_type:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        handle.write(chunk)
            logger.info(f"PDF downloaded successfully and saved to {output_path}")
        else:
            logger.warn(f"Unexpected Content-Type: {content_type}")
            logger.warn("The downloaded file is not a PDF. It might be an error page.")
    else:
        logger.error(f"Failed to download PDF. Status code: {response.status_code}")
        # logger.error(f"Response Headers: {response.headers}")
        # logger.error("Response Content (first 500 characters):")
        # logger.error(response.text[:500])

def create_example_index_file(index_file: Path):
    content = """
This is an example index file for papiry. Any format is allowed! (but Markdown is recommended)

It works like this:
- If a bullet point contains something in square brackets, it is a paper. It should contain a download URL.
- If a section name contains something in square brackets, any papers inside of it will be categorized into a corresponding subdirectory. 

Have fun!

# [ModelBasedRL] Model-based RL

- [AlphaZero] Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm: https://arxiv.org/abs/1712.01815
- [PlaNet] Learning Latent Dynamics for Planning from Pixels: https://arxiv.org/abs/1811.04551
- [MuZero] Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model: https://arxiv.org/abs/1911.08265

# [NTP] Neural Theorem Proving

- [HTPS] HyperTree Proof Search for Neural Theorem Proving: https://openreview.net/pdf?id=J4pX8Q8cxHH + https://openreview.net/attachment?id=J4pX8Q8cxHH&name=supplementary_material
- [AlphaGeometry1] Solving olympiad geometry without human demonstrations: https://www.nature.com/articles/s41586-023-06747-5.pdf
""".lstrip()
    with open(index_file, "w") as f:
        f.write(content)

def search_index(index: Index, query: str) -> list[Paper]:
    def normalize(s: str) -> str:
        return s.lower()

    query = normalize(query)
    results = []
    for paper in index.papers:
        if query in normalize(paper.get_title()) or query in normalize(paper.filename):
            results.append(paper)
    return results

def run_download(args):
    if not args.out_dir.exists():
        logger.error(f"Output directory does not exist: {args.out_dir}")
        return

    index = Index.read_index(args.index)
    existing = read_existing(args.out_dir)
    logger.info(f"Found {len(index.papers)} papers in {args.index}, will download missing ones to {args.out_dir}...")
    download_index(index, existing, args.out_dir)


def run_search(args):
    index = Index.read_index(args.index)
    results = search_index(index, args.query)
    for i, result in enumerate(results):
        logger.info(f"{i + 1}. {result.get_path()}: {result.get_title()} ({", ".join(result.links)})")

def run_list(args):
    index = Index.read_index(args.index)
    for paper in index.papers:
        logger.info(f"{paper.get_path()}: {paper.get_title()} ({", ".join(paper.links)})")


def main():
    """Entry point for the papiry command."""
    args = get_parser().parse_args()

    if not args.index.exists():
        logger.warn(f"Index file does not exist: {args.index}")
        if not args.index.parent.exists():
            logger.error(f"Won't create the index file since the directory does not exist: {args.index.parent}")
            return
        logger.info(f"Creating example index file... Run `papiry` again to download the papers.")
        create_example_index_file(args.index)
        return

    if args.command == "download":
        run_download(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "list":
        run_list(args)
    else:
        logger.error(f"Unknown command: {args.command}")

if __name__ == "__main__":
    main()
