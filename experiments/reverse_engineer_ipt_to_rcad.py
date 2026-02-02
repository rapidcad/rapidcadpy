"""
Reverse engineer IPT files to Python code using RapidCADpy.
"""

import argparse
import os
import glob
import sys
from pathlib import Path

# Add project root to path
# Use insert(0, ...) to ensure local package takes precedence over installed ones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rapidcadpy
print(f"Using rapidcadpy from: {rapidcadpy.__file__}")

from rapidcadpy import InventorApp
from rapidcadpy.integrations.inventor.reverse_engineer import InventorReverseEngineer


def get_unique_filename(output_file):
    """
    Get a unique filename by appending _1, _2, etc. if file already exists.
    
    Args:
        output_file: Desired output file path
        
    Returns:
        Unique file path that doesn't exist yet
    """
    if not os.path.exists(output_file):
        return output_file
    
    # Split into directory, base name, and extension
    dir_name = os.path.dirname(output_file)
    base_name = os.path.basename(output_file)
    name_without_ext, ext = os.path.splitext(base_name)
    
    # Try appending _1, _2, _3, etc.
    counter = 1
    while True:
        new_name = f"{name_without_ext}_{counter}{ext}"
        new_path = os.path.join(dir_name, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1


def reverse_engineer_ipt_file(file_path, output_file, overwrite=False):
    """
    Reverse engineer an IPT file to Python code.

    Args:
        app: InventorApp instance (reused for efficiency)
        file_path: Path to the IPT file
        output_file: Path to save the generated code
        overwrite: Whether to overwrite existing files

    Returns:
        Generated Python code as string
    """
    app = InventorApp(headless=True)
    try:
        # Open document
        doc = app.open_document(file_path)

        # Create reverse engineer instance
        engineer = InventorReverseEngineer(doc)

        # Generate code
        generated_code = engineer.analyze_ipt_file()

        # Get unique filename if file already exists
        if overwrite:
            unique_output_file = output_file
        else:
            unique_output_file = get_unique_filename(output_file)
        
        # Save to file
        output_dir = os.path.dirname(unique_output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        with open(unique_output_file, "w") as f:
            f.write(generated_code)
        
        if unique_output_file != output_file:
            print(f"✓ Generated: {unique_output_file} (renamed to avoid overwrite)")
        else:
            print(f"✓ Generated: {unique_output_file}")

        # Close document
        doc.Close(SkipSave=True)

        return generated_code
    except Exception as e:
        print(f"✗ Error processing {file_path}: {e}")
        return None


def reverse_engineer_directory(input_dir, output_dir, recursive=False, overwrite=False):
    """
    Reverse engineer all IPT files in a directory.

    Args:
        input_dir: Directory containing IPT files
        output_dir: Directory to save generated Python files
        recursive: Whether to search subdirectories recursively
        overwrite: Whether to overwrite existing files
    """
    # Initialize Inventor once for all files
    app = InventorApp(headless=True)
    
    # Find all IPT files
    if recursive:
        ipt_files = glob.glob(os.path.join(input_dir, "**", "*.ipt"), recursive=True)
    else:
        ipt_files = glob.glob(os.path.join(input_dir, "*.ipt"))
    
    if not ipt_files:
        print(f"No IPT files found in {input_dir}")
        return
    
    print(f"Found {len(ipt_files)} IPT file(s) to process")
    
    success_count = 0
    failure_count = 0
    
    for ipt_file in ipt_files:
        # Create corresponding output path
        rel_path = os.path.relpath(ipt_file, input_dir)
        output_file = os.path.join(output_dir, rel_path).replace('.ipt', '.py')
        
        # Process file
        result = reverse_engineer_ipt_file(ipt_file, output_file, overwrite=overwrite)
        
        if result is not None:
            success_count += 1
        else:
            failure_count += 1
    
    print(f"\n{'='*60}")
    print(f"Processing complete:")
    print(f"  Success: {success_count}")
    print(f"  Failures: {failure_count}")
    print(f"  Total: {len(ipt_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reverse engineer IPT files to Python code.")
    parser.add_argument("--file", help="Path to a single IPT file to process")
    parser.add_argument("--output", help="Path to the output Python file (used with --file)")
    parser.add_argument("--dir", help="Directory containing IPT files to process")
    parser.add_argument("--outdir", help="Output directory for generated Python files (used with --dir)")
    parser.add_argument("--recursive", action="store_true", help="Recursively search for IPT files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files instead of creating new ones")
    
    args = parser.parse_args()
    
    if args.file and args.output:
        # Process single file
        if not os.path.exists(args.file):
            print(f"Error: Input file does not exist: {args.file}")
        else:
            # Convert to absolute path
            abs_file_path = os.path.abspath(args.file)
            reverse_engineer_ipt_file(abs_file_path, args.output, overwrite=args.overwrite)
            
    elif args.dir and args.outdir:
        # Process directory
        if not os.path.exists(args.dir):
            print(f"Error: Input directory does not exist: {args.dir}")
        else:
            # Convert to absolute path
            abs_dir_path = os.path.abspath(args.dir)
            reverse_engineer_directory(abs_dir_path, args.outdir, args.recursive, overwrite=args.overwrite)
            
    else:
        # Fallback to hardcoded testing paths if no args provided (or print usage)
        print("No arguments provided. Using default test paths...")
        # Keeping existing test code as fallback for now, or just print help
        # parser.print_help()
        
        # Default test logic (modified to use relative paths if files match)
        input_dir = "C:\\Users\\Administrator\\Documents\\sample_2025_11_18_11_16\\1"
        output_dir = "C:\\Users\\Administrator\\Documents\\shaft_llm_data\\1"
        # Try to use a file we know exists in the workspace for testing behavior
        # test_file = "tests/test_files/1.ipt"
        reverse_engineer_directory(input_dir, output_dir, recursive=True) 
        #if os.path.exists(test_file):
        #     reverse_engineer_ipt_file(os.path.abspath(test_file), "99.py")
        #else:
        #    parser.print_help()