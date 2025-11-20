#!/usr/bin/env python3
"""
Convert PAPER_DISCREPANCY_ANALYSIS.md to HTML
"""

import subprocess
import sys
from pathlib import Path

def convert_to_html():
    """Convert markdown to HTML"""
    input_file = "PAPER_DISCREPANCY_ANALYSIS.md"
    output_file = "PAPER_DISCREPANCY_ANALYSIS.html"
    
    if not Path(input_file).exists():
        print(f"❌ Input file {input_file} not found")
        return False
    
    try:
        cmd = [
            "pandoc",
            input_file,
            "-o", output_file,
            "--standalone",
            "--toc",
            "--number-sections",
            "--metadata", "title=Paper Discrepancy Analysis",
            "--css", "https://cdn.jsdelivr.net/npm/github-markdown-css@5/github-markdown-light.css",
            "--template", "html5"
        ]
        
        print("🔄 Converting to HTML using pandoc...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Successfully converted to {output_file}")
            print("💡 You can print this HTML to PDF using your browser:")
            print(f"   1. Open {output_file} in your browser")
            print("   2. Press Cmd+P (Mac) or Ctrl+P (Windows)")
            print("   3. Select 'Save as PDF'")
            return True
        else:
            print(f"❌ Pandoc HTML failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ Pandoc not found. Please install pandoc:")
        print("   macOS: brew install pandoc")
        print("   Ubuntu: sudo apt install pandoc")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("📄 PAPER DISCREPANCY ANALYSIS - HTML CONVERSION")
    print("=" * 60)
    
    success = convert_to_html()
    
    if success:
        print("\n🎯 CONVERSION COMPLETE!")
        print("📊 Files available:")
        print("   - PAPER_DISCREPANCY_ANALYSIS.md (source)")
        print("   - PAPER_DISCREPANCY_ANALYSIS.html (output)")
        print("   - complete_figure3_reproduction.png (embedded)")
        print("   - complete_figure3_results.json (referenced)")
    else:
        print("\n❌ CONVERSION FAILED!")

if __name__ == "__main__":
    main()
