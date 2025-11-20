#!/usr/bin/env python3
"""
Script to download and convert arXiv PDF to text
"""
import requests
from pypdf import PdfReader
import io
import sys

def download_and_convert_arxiv_pdf(url, output_path=None):
    """Download PDF from URL and convert to text"""
    try:
        print(f"Downloading PDF from {url}...")
        
        # Download the PDF
        response = requests.get(url)
        response.raise_for_status()
        
        # Create a PDF reader from the downloaded content
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        # Extract text from all pages
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
            text += "\n"
        
        # Determine output path
        if output_path is None:
            output_path = "arxiv_paper.txt"
        
        # Write text to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Successfully converted PDF to {output_path}")
        print(f"Total pages: {len(reader.pages)}")
        return output_path
        
    except Exception as e:
        print(f"Error downloading/converting PDF: {str(e)}")
        return None

def main():
    url = "https://arxiv.org/pdf/2406.02528"
    output_file = download_and_convert_arxiv_pdf(url, "matmul_free_paper.txt")
    
    if output_file:
        # Show first few lines
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"\nFirst 10 lines of converted text:")
            for i, line in enumerate(lines[:10]):
                print(f"{i+1}: {line.strip()}")

if __name__ == "__main__":
    main()
