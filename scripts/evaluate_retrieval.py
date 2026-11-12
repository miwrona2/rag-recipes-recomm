#!/usr/bin/env python3
"""
CLI Script for evaluating RAG retrieval performance (Recall@k and Precision@k).
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.retrieval_evaluation import (
    evaluate_retrieval_dataset,
    format_evaluation_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate Precision@k and Recall@k for RAG recipe retrieval."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=str(PROJECT_ROOT / "app" / "data" / "sample.json"),
        help="Path to evaluation dataset JSON file (default: app/data/sample.json)",
    )
    parser.add_argument(
        "-k",
        "--k-values",
        nargs="+",
        type=int,
        default=[3, 5, 10, 20],
        help="List of k values for evaluation (default: 3 5 10 20)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.35,
        help="Vector search cosine similarity match threshold (default: 0.35)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to output JSON file with metrics results",
    )
    return parser.parse_args()


async def main() -> int:
    args = parse_args()
    dataset_path = Path(args.dataset)

    if not dataset_path.exists():
        print(f"[ERROR] Plik zbioru danych nie istnieje: {dataset_path}", file=sys.stderr)
        return 1

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except Exception as e:
        print(f"[ERROR] Błąd podczas wczytywania pliku JSON: {e}", file=sys.stderr)
        return 1

    if not isinstance(dataset, list):
        print("[ERROR] Zbiór danych musi być listą obiektów ewaluacyjnych.", file=sys.stderr)
        return 1

    k_values: List[int] = sorted(list(set(args.k_values)))
    print(
        f"Rozpoczynanie ewaluacji dla {len(dataset)} próbek testowych (k={k_values}, threshold={args.threshold})...\n"
    )

    try:
        results = await evaluate_retrieval_dataset(
            dataset=dataset,
            k_values=k_values,
            match_threshold=args.threshold,
        )
    except Exception as e:
        print(f"[ERROR] Błąd podczas ewaluacji: {e}", file=sys.stderr)
        return 1

    summary = format_evaluation_summary(
        results=results,
        k_values=k_values,
        total_samples=len(dataset),
        dataset_name=str(dataset_path.name),
    )
    print(summary)

    if args.output:
        output_path = Path(args.output)
        try:
            output_payload = {
                "dataset": str(dataset_path),
                "totalSamples": len(dataset),
                "kValues": k_values,
                "matchThreshold": args.threshold,
                "metrics": results,
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_payload, f, indent=2, ensure_ascii=False)
            print(f"\nWyniki zapisano pomyślnie do: {output_path}")
        except Exception as e:
            print(f"[ERROR] Nie udało się zapisać wyników do pliku {output_path}: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
