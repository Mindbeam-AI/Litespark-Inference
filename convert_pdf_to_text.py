#!/usr/bin/env python3
"""
Script to convert PDF files to text format
"""
import sys
from pypdf import PdfReader
import os

def convert_pdf_to_text(pdf_path, output_path=None):
    """Convert a PDF file to text"""
    try:
        # Create a PDF reader object
        reader = PdfReader(pdf_path)
        
        # Extract text from all pages
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"\n--- Page {page_num + 1} ---\n"
            text += page.extract_text()
            text += "\n"
        
        # Determine output path
        if output_path is None:
            base_name = os.path.splitext(pdf_path)[0]
            output_path = f"{base_name}.txt"
        
        # Write text to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"Successfully converted {pdf_path} to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error converting {pdf_path}: {str(e)}")
        return None

def main():
    # Convert both PDF files in the current directory
    pdf_files = [
        "matmulFree_v1-report.pdf",
        "training_results_step9000_20251114_001031.pdf"
    ]
    
    for pdf_file in pdf_files:
        if os.path.exists(pdf_file):
            print(f"Converting {pdf_file}...")
            convert_pdf_to_text(pdf_file)
        else:
            print(f"File not found: {pdf_file}")

if __name__ == "__main__":
    main()
