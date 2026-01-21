"""
Driver script to run subgraphx explanations with varying num_top_edges values
and track fidelity and sparsity metrics.
"""

import subprocess
import re
import sys
import numpy as np

# num_top_edges values to test
START = 1
END = 40
VALUES = 7
num_top_edges_values = np.round(np.linspace(START, END, num=VALUES)).astype(int).tolist()

def parse_metrics(output):
    """Parse fidelity_node_gnn_prob+ and node_mask_sparsity from command output."""
    pattern = r'raw\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
    match = re.search(pattern, output)
    if match:
        # Extract the relevant values (indices based on column order)
        fidelity = float(match.group(1))  # fidelity_node_gnn_prob+
        sparsity = float(match.group(3))  # node_mask_sparsity
        return fidelity, sparsity
    return None, None

def main():
    results = []

    for x in num_top_edges_values:
        cmd = [
            "python3", "code/main.py",
            "--dataset_name", "bbbp",
            "--model_name", "gin",
            "--explainer_name", "subgraphx",
            "--num_explained_y", "100",
            "--num_top_edges", str(x),
            "--mask_transformation", "None",
            "--mask_save_dir", "None",
            "--pred_type", "correct",
            "--focus", "model",
            "--paper_eval"
        ]

        print(f"\n{'='*60}")
        print(f"Running with --num_top_edges = {x}")
        print(f"{'='*60}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            output = result.stdout + result.stderr

            fidelity, sparsity = parse_metrics(output)

            if fidelity is not None and sparsity is not None:
                print(f"✓ fidelity_node_gnn_prob+: {fidelity}")
                print(f"✓ node_mask_sparsity: {sparsity}")
                results.append({
                    'num_top_edges': x,
                    'fidelity': fidelity,
                    'sparsity': sparsity
                })
            else:
                print("✗ Failed to parse metrics from output")
                print("Last 50 lines of output:")
                for line in output.split('\n')[-50:]:
                    print(line)
                results.append({
                    'num_top_edges': x,
                    'fidelity': 'FAILED',
                    'sparsity': 'FAILED'
                })

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout after 1 hour for num_top_edges = {x}")
            results.append({
                'num_top_edges': x,
                'fidelity': 'TIMEOUT',
                'sparsity': 'TIMEOUT'
            })
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                'num_top_edges': x,
                'fidelity': 'ERROR',
                'sparsity': 'ERROR'
            })

    # Print summary table
    print(f"\n{'='*70}")
    print("SubgraphX Summary Table")
    print(f"{'='*70}")
    print(f"{'num_top_edges':<15} {'fidelity_node_gnn_prob+':<25} {'node_mask_sparsity':<20}")
    print("-" * 70)

    for r in results:
        fidelity_str = f"{r['fidelity']:.4f}" if isinstance(r['fidelity'], float) else r['fidelity']
        sparsity_str = f"{r['sparsity']:.4f}" if isinstance(r['sparsity'], float) else r['sparsity']
        print(f"{r['num_top_edges']:<15} {fidelity_str:<25} {sparsity_str:<20}")

    print("-" * 70)

if __name__ == "__main__":
    main()
