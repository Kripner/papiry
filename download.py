import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import requests
from pypdf import PdfReader, PdfWriter

parser = argparse.ArgumentParser()
parser.add_argument("--index_file", type=Path, default="index.md")
parser.add_argument("--output_dir", type=Path, default="pdf")


# TODO: symlinks

@dataclass
class Paper:
    filename: str
    section: str
    links: list[str]


def read_index(index_file: Path) -> list[Paper]:
    index = []
    with open(index_file, "r") as f:
        section = ""
        for i, line in enumerate(f):
            if line.startswith("%"):
                continue

            if line.startswith("#"):
                section = find_inside_brackets(line[1:])
            elif line.startswith("-"):
                filename = find_inside_brackets(line[1:])
                if filename is None:
                    print(f"WARN: Cannot extract paper filename from line {i + 1} in {index_file}")
                    continue
                urls = find_urls(line[1:])
                index.append(Paper(filename, section, urls))
    return index


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
    # print("\n".join([f"{k} -> {v}" for k,v in existing.items()]))


def download_index(index: list[Paper], existing: dict[str, Path], output_dir: Path):
    for paper in index:
        output_path = output_dir / paper.section / (paper.filename + ".pdf")
        if output_path.exists():
            continue
        if paper.filename in existing:
            print(f"Moving {existing[paper.filename]} to {output_path}...")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            existing[paper.filename].rename(output_path)
            continue
        if len(paper.links) == 0:
            print(f"WARN: No URL for paper {paper.filename} in section {paper.section}")
            continue
        pdf_urls = [get_pdf_url(url) for url in paper.links]
        pdf_urls = [u for u in pdf_urls if u is not None]
        if len(pdf_urls) == 0:
            print(f"Cannot resolve any download URL for {paper.filename} from paper URLs: {", ".join(paper.links)}")
            continue
        download_and_merge_pdfs(pdf_urls, output_path)


def get_pdf_url(paper_url: str) -> str | None:
    if (paper_url.endswith(".pdf")
            or paper_url.startswith("https://openreview.net/pdf")
            or paper_url.startswith("https://openreview.net/attachment")
    ):
        return paper_url
    if paper_url.startswith("https://arxiv.org/abs/"):
        return paper_url.replace("/abs/", "/pdf/")
    if paper_url.startswith("https://arxiv.org/pdf/"):
        return paper_url
    if paper_url.startswith("https://www.nature.com/articles/"):
        return paper_url + ".pdf"
    if paper_url.startswith("https://openreview.net/forum?id="):
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

    print(f"Merging {[p.name for p in parts_paths]} -> {output_path.name}")
    merge_pdfs(output_path, parts_paths)

    for part_path in parts_paths:
        part_path.unlink()


def merge_pdfs(output_path, pdf_paths):
    writer = PdfWriter()

    for pdf_path in pdf_paths:
        print(pdf_path)
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            writer.add_page(page)

    with open(output_path, "wb") as output_file:
        writer.write(output_file)


def download_pdf(pdf_url: str, output_path: Path):
    print(f"Downloading {pdf_url} to {output_path}...")
    session = requests.Session()
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
            print(f"PDF downloaded successfully and saved to {output_path}")
        else:
            print(f"Unexpected Content-Type: {content_type}")
            print("The downloaded file is not a PDF. It might be an error page.")
    else:
        print(f"Failed to download PDF. Status code: {response.status_code}")
        print("Response Headers:", response.headers)
        print("Response Content (first 500 characters):")
        print(response.text[:500])


def main(args):
    output_dir = (Path(__file__).parent / args.output_dir).resolve()
    index_file = (Path(__file__).parent / args.index_file).resolve()

    index = read_index(index_file)
    existing = read_existing(output_dir)
    print(f"Found {len(index)} papers in {index_file}, will download missing ones to {output_dir}...")
    download_index(index, existing, output_dir)


if __name__ == "__main__":
    main(parser.parse_args())
