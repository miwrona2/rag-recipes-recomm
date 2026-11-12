import asyncio
import json
import os
from typing import Any, Callable, Dict, List, Optional, Set, Union
from app.core.logging import logger
from app.services.candidate_filter import filter_excluded_candidates
from app.services.vector_search import retrieve_candidates


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Precision@k: proportion of top-k retrieved items that are relevant.

    :param retrieved_ids: Ordered list of retrieved candidate recipe IDs.
    :param relevant_ids: Set of ground-truth relevant recipe IDs.
    :param k: Cutoff rank.
    :return: Precision score at rank k in range [0.0, 1.0].
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant_retrieved = sum(1 for item_id in top_k if str(item_id) in relevant_ids)
    return relevant_retrieved / min(k, len(top_k))


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Calculate Recall@k: proportion of relevant items found in top-k retrieved items.

    :param retrieved_ids: Ordered list of retrieved candidate recipe IDs.
    :param relevant_ids: Set of ground-truth relevant recipe IDs.
    :param k: Cutoff rank.
    :return: Recall score at rank k in range [0.0, 1.0].
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for item_id in top_k if str(item_id) in relevant_ids)
    return relevant_retrieved / len(relevant_ids)


def extract_candidate_ids(candidates: List[Union[Dict[str, Any], Any]]) -> List[str]:
    """
    Extract uniform string recipe IDs from candidate dictionaries or candidate objects.

    :param candidates: List of candidate items.
    :return: List of string candidate IDs.
    """
    ids: List[str] = []
    for cand in candidates:
        if isinstance(cand, dict):
            cand_id = cand.get("recipe_id") or cand.get("id") or cand.get("candidateId")
        else:
            cand_id = (
                getattr(cand, "recipe_id", None)
                or getattr(cand, "id", None)
                or getattr(cand, "candidateId", None)
            )
        if cand_id is not None:
            ids.append(str(cand_id))
    return ids


async def evaluate_retrieval_dataset(
    dataset: List[Dict[str, Any]],
    k_values: Optional[List[int]] = None,
    retrieve_fn: Optional[Callable[..., Any]] = None,
    filter_fn: Optional[Callable[..., Any]] = None,
    match_threshold: float = 0.35,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Evaluate retrieval quality across a benchmark dataset for specified k cutoffs.

    :param dataset: List of benchmark evaluation samples containing input and relevant recipe IDs.
    :param k_values: List of k cutoffs to compute metrics for (default: [3, 5, 10, 20]).
    :param retrieve_fn: Optional retrieval candidate provider function.
    :param filter_fn: Optional candidate exclusion filter function.
    :param match_threshold: Cosine similarity threshold for vector search.
    :param kwargs: Additional arguments passed to retrieval function.
    :return: Dictionary containing Mean Precision@k and Mean Recall@k scores.
    """
    if k_values is None:
        k_values = [3, 5, 10, 20]

    total_samples = len(dataset)
    metrics_sum: Dict[str, float] = {f"precision@{k}": 0.0 for k in k_values}
    metrics_sum.update({f"recall@{k}": 0.0 for k in k_values})

    if total_samples == 0:
        return metrics_sum

    retrieve = retrieve_fn or retrieve_candidates
    filter_candidates = filter_fn or filter_excluded_candidates
    max_k = max(k_values) if k_values else 20

    for sample in dataset:
        sample_input = sample.get("input", {})
        mood = sample_input.get("mood", "")
        meal_type = sample_input.get("mealType") or sample_input.get("meal_type", "")
        exclude_urls = sample_input.get("excludeUrls") or sample_input.get("exclude_urls", [])

        relevant_raw = (
            sample.get("relevantRecipeIds")
            or sample.get("relevant_recipe_ids")
            or []
        )
        relevant_ids = set(map(str, relevant_raw))

        # 1. Fetch candidates from vector database
        candidates = await retrieve(
            mood=mood,
            meal_type=meal_type,
            match_count=max_k,
            match_threshold=match_threshold,
            **kwargs,
        )

        # 2. Filter excluded URLs and IDs
        filtered_candidates = filter_candidates(
            candidates,
            exclude_urls=exclude_urls,
        )

        retrieved_ids = extract_candidate_ids(filtered_candidates)

        # 3. Calculate metrics per sample for each k
        for k in k_values:
            p_k = calculate_precision_at_k(retrieved_ids, relevant_ids, k)
            r_k = calculate_recall_at_k(retrieved_ids, relevant_ids, k)
            metrics_sum[f"precision@{k}"] += p_k
            metrics_sum[f"recall@{k}"] += r_k

    # 4. Aggregated means (Mean Precision@k and Mean Recall@k)
    results = {
        metric: total / total_samples for metric, total in metrics_sum.items()
    }
    return results


def format_evaluation_summary(
    results: Dict[str, float],
    k_values: List[int],
    total_samples: int,
    dataset_name: Optional[str] = None,
) -> str:
    """
    Format evaluation results into a human-readable console report.

    :param results: Dictionary mapping metric names to mean values.
    :param k_values: Evaluated cutoff levels.
    :param total_samples: Total number of evaluated dataset samples.
    :param dataset_name: Optional label/path for the evaluated dataset.
    :return: Formatted summary string.
    """
    lines: List[str] = []
    lines.append("=" * 76)
    lines.append("           RAPORT EWALUACJI FAZY RETRIEVALU (RAG)           ")
    lines.append("=" * 76)

    if dataset_name:
        lines.append(f"Zbiór danych : {dataset_name}")
    lines.append(f"Liczba próbek: {total_samples}")
    lines.append(f"Wartości k   : {', '.join(map(str, k_values))}")
    lines.append("-" * 76)
    lines.append(f"{'Metryka':<16} | {'Wartość':<12} | {'Procent':<10} | {'Interpretacja'}")
    lines.append("-" * 76)

    for k in k_values:
        p_val = results.get(f"precision@{k}", 0.0)
        p_pct = f"{p_val * 100:.2f}%"
        lines.append(
            f"{f'Precision@{k}':<16} | {p_val:<12.4f} | {p_pct:<10} | "
            f"Odsetek trafnych przepisów w top-{k}"
        )

    lines.append("-" * 76)

    for k in k_values:
        r_val = results.get(f"recall@{k}", 0.0)
        r_pct = f"{r_val * 100:.2f}%"
        lines.append(
            f"{f'Recall@{k}':<16} | {r_val:<12.4f} | {r_pct:<10} | "
            f"Pokrycie relewantnych przepisów w top-{k}"
        )

    lines.append("=" * 76)
    lines.append("OPIS METRYK I WSKAZÓWKI DO OPTYMALIZACJI:")
    lines.append(
        " • Precision@k (Precyzja): Mierzy jakość zwróconych wyników - jaki odsetek\n"
        "   spośród k przepisów jest faktycznie dopasowany do preferencji użytkownika.\n"
        "   Wyższa precyzja ogranicza szum informacyjny przekazywany do promptu LLM."
    )
    lines.append(
        " • Recall@k (Pokrycie / Pełność): Mierzy zdolność systemu do odnalezienia\n"
        "   wszystkich wzorcowych przepisów w pierwszych k wynikach.\n"
        "   Wyższy Recall gwarantuje, że kluczowe przepisy dotrą do kolejnych etapów RAG."
    )
    lines.append(
        " • Balans k (Trade-off): Zwiększanie wartości k podnosi Recall kosztem Precision\n"
        "   oraz zwiększa liczbę tokenów w kontekście LLM. Optymalne k maksymalizuje\n"
        "   Recall przy zachowaniu wysokiej czystości wyników (Precision)."
    )
    lines.append("=" * 76)

    return "\n".join(lines)
