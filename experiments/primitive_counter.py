"""
Primitive Counter Script

Analyzes Python files in a directory to count CAD primitives and classify shapes.

Rules:
- A CAD object is a cylinder if it contains exactly 1 circle, 1 extrude, and 0 line_to operations
- A CAD object is a box if it contains exactly 4 line_to, 1 extrude, and 0 circle operations
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, Tuple


class PrimitiveCounter:
    """Counts CAD primitives in Python files and classifies shapes."""

    def __init__(self):
        self.total_files = 0
        self.total_circles = 0
        self.total_line_to = 0
        self.total_extrudes = 0
        self.total_boxes = 0
        self.total_cylinders = 0
        self.total_other = 0
        self.file_results = []

    def count_primitives(self, file_path: Path) -> Dict[str, int]:
        """
        Count occurrences of circle, line_to, and extrude in a Python file.

        Args:
            file_path: Path to the Python file

        Returns:
            Dictionary with counts of each primitive
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Count occurrences - match as methods (with parentheses) or standalone words
            # Look for .circle( or circle( to match method calls
            circles = len(
                re.findall(
                    r"\.circle\s*\(|^circle\s*\(", content, re.MULTILINE | re.IGNORECASE
                )
            )
            line_tos = len(
                re.findall(
                    r"\.line_to\s*\(|^line_to\s*\(",
                    content,
                    re.MULTILINE | re.IGNORECASE,
                )
            )
            extrudes = len(
                re.findall(
                    r"\.extrude\s*\(|^extrude\s*\(",
                    content,
                    re.MULTILINE | re.IGNORECASE,
                )
            )

            return {"circle": circles, "line_to": line_tos, "extrude": extrudes}
        except Exception as e:
            print(f"Warning: Could not read {file_path}: {e}")
            return {"circle": 0, "line_to": 0, "extrude": 0}

    def classify_shape(self, counts: Dict[str, int]) -> str:
        """
        Classify the shape based on primitive counts.

        Args:
            counts: Dictionary with primitive counts

        Returns:
            Shape type: 'box', 'cylinder', or 'other'
        """
        circles = counts["circle"]
        line_tos = counts["line_to"]
        extrudes = counts["extrude"]

        # Cylinder: exactly 1 circle, 1 extrude, 0 line_to
        if circles == 1 and extrudes == 1 and line_tos == 0:
            return "cylinder"

        # Box: exactly 4 line_to, 1 extrude, 0 circle
        if line_tos == 4 and extrudes == 1 and circles == 0:
            return "box"

        return "other"

    def process_file(self, file_path: Path):
        """Process a single Python file."""
        counts = self.count_primitives(file_path)
        shape = self.classify_shape(counts)

        # Update totals
        self.total_files += 1
        self.total_circles += counts["circle"]
        self.total_line_to += counts["line_to"]
        self.total_extrudes += counts["extrude"]

        if shape == "box":
            self.total_boxes += 1
        elif shape == "cylinder":
            self.total_cylinders += 1
        else:
            self.total_other += 1

        # Store result
        self.file_results.append(
            {"file": file_path.name, "counts": counts, "shape": shape}
        )

    def process_directory(self, directory: Path, verbose: bool = False):
        """
        Process all Python files in a directory.

        Args:
            directory: Path to directory containing Python files
            verbose: If True, print details for each file
        """
        if not directory.exists():
            print(f"Error: Directory not found: {directory}")
            sys.exit(1)

        if not directory.is_dir():
            print(f"Error: Path is not a directory: {directory}")
            sys.exit(1)

        # Find all .py files
        py_files = list(directory.glob("*.py"))

        if not py_files:
            print(f"Warning: No Python files found in {directory}")
            return

        print(f"Found {len(py_files)} Python file(s) in {directory}")
        print("-" * 60)

        # Process each file
        for py_file in sorted(py_files):
            self.process_file(py_file)

            if verbose:
                result = self.file_results[-1]
                print(f"{result['file']}: {result['counts']} -> {result['shape']}")

        if verbose:
            print("-" * 60)

    def print_summary(self):
        """Print summary statistics."""
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)

        print(f"\nTotal files processed: {self.total_files}")

        print(f"\nPrimitive counts:")
        print(f"  Circles:  {self.total_circles}")
        print(f"  Line_to:  {self.total_line_to}")
        print(f"  Extrudes: {self.total_extrudes}")

        # Average extrudes per file
        if self.total_files > 0:
            avg_extrudes = self.total_extrudes / self.total_files
        else:
            avg_extrudes = 0.0
        print(f"\nAverage extrudes per file: {avg_extrudes:.2f}")

        print(f"\nShape classification:")
        print(f"  Boxes:     {self.total_boxes}")
        print(f"  Cylinders: {self.total_cylinders}")
        print(f"  Other:     {self.total_other}")

        if self.total_files > 0:
            box_pct = (self.total_boxes / self.total_files) * 100
            cylinder_pct = (self.total_cylinders / self.total_files) * 100
            other_pct = (self.total_other / self.total_files) * 100

            print(f"\nPercentages:")
            print(f"  Boxes:     {box_pct:.2f}%")
            print(f"  Cylinders: {cylinder_pct:.2f}%")
            print(f"  Other:     {other_pct:.2f}%")

        print("=" * 60)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python primitive_counter.py <directory> [--verbose]")
        print("\nAnalyzes Python files to count CAD primitives and classify shapes.")
        print("\nOptions:")
        print("  --verbose    Print details for each file")
        sys.exit(1)

    directory = Path(sys.argv[1])
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    counter = PrimitiveCounter()
    counter.process_directory(directory, verbose=verbose)
    counter.print_summary()


if __name__ == "__main__":
    main()
