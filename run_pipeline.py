#!/usr/bin/env python3
"""
ArchIntel Pipeline Runner
Runs the complete pipeline from fetching papers to building the index.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    """Run the complete ArchIntel pipeline."""
    print("🏗️ ArchIntel Pipeline Runner")
    print("This will run the complete pipeline from fetching papers to building the index.")
    
    # Check if we're in the right directory
    if not Path("scripts").exists():
        print("❌ Error: Please run this script from the ArchIntel-Architecture-Intelligence directory")
        sys.exit(1)
    
    # Step 1: Fetch papers
    if not run_command("python3 scripts/fetch_papers.py", "Step 1: Fetching papers"):
        print("\n❌ Failed to fetch papers. Check your internet connection and seeds/papers.yaml")
        sys.exit(1)
    
    # Step 2: Parse PDFs
    if not run_command("python3 scripts/parse_pdf.py", "Step 2: Parsing PDFs"):
        print("\n❌ Failed to parse PDFs. Check that PDFs were downloaded successfully")
        sys.exit(1)
    
    # Step 3: Build index
    if not run_command("python3 scripts/build_index.py", "Step 3: Building search index"):
        print("\n❌ Failed to build index. Check that text extraction was successful")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("🎉 Pipeline completed successfully!")
    print(f"{'='*60}")
    print("\nYou can now start querying your papers:")
    print("  • CLI: python3 scripts/query_index.py")
    print("  • Web: streamlit run app.py")
    print("\nHappy researching! 🎓")

if __name__ == "__main__":
    main()
