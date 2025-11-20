#!/usr/bin/env python3
"""
Convert PAPER_DISCREPANCY_ANALYSIS.md to PDF
"""

import subprocess
import sys
from pathlib import Path

def convert_to_pdf():
    """Convert markdown to PDF using pandoc"""
    
    input_file = "PAPER_DISCREPANCY_ANALYSIS.md"
    output_file = "PAPER_DISCREPANCY_ANALYSIS.pdf"
    
    if not Path(input_file).exists():
        print(f"❌ Input file {input_file} not found")
        return False
    
    try:
        # Try pandoc with different engines
        engines = ["pdflatex", "xelatex", "lualatex"]

        for engine in engines:
            cmd = [
                "pandoc",
                input_file,
                "-o", output_file,
                f"--pdf-engine={engine}",
                "--variable", "geometry:margin=1in",
                "--variable", "fontsize=11pt",
                "--variable", "documentclass=article",
                "--toc",
                "--number-sections"
            ]

            print(f"🔄 Trying PDF engine: {engine}...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Successfully converted using {engine}")
                return True
            else:
                print(f"❌ {engine} failed: {result.stderr.strip()}")
                continue
        

        print("❌ All PDF engines failed")
        return False
            
    except FileNotFoundError:
        print("❌ Pandoc not found. Please install pandoc and xelatex:")
        print("   macOS: brew install pandoc basictex")
        print("   Ubuntu: sudo apt install pandoc texlive-xetex")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("📄 PAPER DISCREPANCY ANALYSIS - PDF CONVERSION")
    print("=" * 60)
    
    success = convert_to_pdf()
    
    if success:
        print("\n🎯 CONVERSION COMPLETE!")
        print("📊 Files generated:")
        print("   - PAPER_DISCREPANCY_ANALYSIS.md (source)")
        print("   - PAPER_DISCREPANCY_ANALYSIS.pdf (output)")
        print("   - complete_figure3_reproduction.png (embedded)")
        print("   - complete_figure3_results.json (referenced)")
    else:
        print("\n❌ CONVERSION FAILED!")
        print("Please install pandoc and try again.")

if __name__ == "__main__":
    main()
