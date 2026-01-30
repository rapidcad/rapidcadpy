#!/bin/bash
# Example usage script for deepcad_to_loadcase.py

# Ensure we're using the cadgpt conda environment
# This script demonstrates various usage patterns

echo "=================================================="
echo "DeepCAD to FEA Load-Case Converter - Examples"
echo "=================================================="

# Example 1: Basic usage with default parameters
echo -e "\n1. Basic usage (default parameters):"
echo "   Command: conda run -n cadgpt python experiments/deepcad_to_loadcase.py \\"
echo "              --step_dir /path/to/steps \\"
echo "              --out_dir /path/to/output"

# Example 2: Custom force and tolerances
echo -e "\n2. Custom force magnitude and tolerances:"
echo "   Command: conda run -n cadgpt python experiments/deepcad_to_loadcase.py \\"
echo "              --step_dir ./tests/test_files \\"
echo "              --out_dir ./output/loadcases \\"
echo "              --force_newtons 1500 \\"
echo "              --face_tol 1.0 \\"
echo "              --box_pad 2.0"

# Example 3: With custom opposing threshold
echo -e "\n3. Adjust opposing face detection threshold:"
echo "   Command: conda run -n cadgpt python experiments/deepcad_to_loadcase.py \\"
echo "              --step_dir ./tests/test_files \\"
echo "              --out_dir ./output/loadcases \\"
echo "              --prefer_opposing 0.90"

# Example 4: Full processing pipeline
echo -e "\n4. Full processing with custom output:"
echo "   Command: conda run -n cadgpt python experiments/deepcad_to_loadcase.py \\"
echo "              --step_dir /mnt/data/deepcad_steps/0000 \\"
echo "              --out_dir /mnt/data/deepcad_loadcases/0000 \\"
echo "              --force_newtons 2000 \\"
echo "              --face_tol 0.8 \\"
echo "              --box_pad 1.5 \\"
echo "              --summary_csv /mnt/data/deepcad_loadcases/0000_summary.csv"

# Example 5: Quick test on current directory
echo -e "\n5. Quick test on current directory STEP files:"
read -p "Run test now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Running test..."
    conda run -n cadgpt python experiments/deepcad_to_loadcase.py \
        --step_dir . \
        --out_dir /tmp/test_loadcases \
        --force_newtons 1000 \
        --face_tol 0.5 \
        --box_pad 1.0
    
    echo -e "\nResults written to: /tmp/test_loadcases/"
    echo "View summary: cat /tmp/test_loadcases/summary.csv"
    echo "View example JSON: cat /tmp/test_loadcases/test_count.json | jq ."
fi

echo -e "\n=================================================="
echo "For more information, see:"
echo "  - experiments/README_deepcad_to_loadcase.md"
echo "  - experiments/deepcad_to_loadcase.py --help"
echo "=================================================="
