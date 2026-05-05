#!/usr/bin/env python3
"""
Generate comprehensive README files for each pipeline stage by reading their docstrings and structure.
"""

import inspect
import importlib.util
import sys
from pathlib import Path

pipeline_stages = [
    ("s1_profiler", "profile_document", 1, "Document Profiler"),
    ("s2_chunkers", "run_all_chunkers", 2, "Chunking Strategy Selector"),
    ("s3_entropy", "refine_boundaries", 3, "Entropy-Based Boundary Refinement"),
    ("s4_boundary", "filter_boundaries", 4, "Boundary Quality Filter"),
    ("s5_graph", "enrich_graph", 5, "Graph Enrichment & Entity Extraction"),
    ("s6_embedding", "embed_chunks", 6, "Contextual Embeddings"),
    ("s7_rl", "run_rl_loop", 7, "RL-Based Hyperparameter Calibration"),
]

pipeline_dir = Path(__file__).parent

for module_name, func_name, stage_num, stage_title in pipeline_stages:
    module_path = pipeline_dir / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        func = getattr(module, func_name)
        docstring = inspect.getdoc(func) or "No docstring"
        print(f"\n{'='*80}")
        print(f"S{stage_num} — {stage_title}")
        print(f"{'='*80}\n")
        print(docstring)
    except Exception as e:
        print(f"Error reading {module_name}: {e}")
