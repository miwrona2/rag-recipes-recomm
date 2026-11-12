import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.retrieval_evaluation import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    extract_candidate_ids,
    evaluate_retrieval_dataset,
    format_evaluation_summary,
)


def test_calculate_precision_at_k_normal():
    retrieved = ["1", "2", "3", "4", "5"]
    relevant = {"1", "3", "7", "9"}

    # Top 3: ["1", "2", "3"] -> 2 relevant ("1", "3") out of 3 = 2/3
    assert pytest.approx(calculate_precision_at_k(retrieved, relevant, 3), 0.001) == 2 / 3

    # Top 5: ["1", "2", "3", "4", "5"] -> 2 relevant out of 5 = 0.4
    assert pytest.approx(calculate_precision_at_k(retrieved, relevant, 5), 0.001) == 0.4

    # Top 1: ["1"] -> 1 relevant out of 1 = 1.0
    assert pytest.approx(calculate_precision_at_k(retrieved, relevant, 1), 0.001) == 1.0


def test_calculate_precision_at_k_edge_cases():
    assert calculate_precision_at_k([], {"1", "2"}, 5) == 0.0
    assert calculate_precision_at_k(["1", "2"], set(), 5) == 0.0
    assert calculate_precision_at_k(["1", "2"], {"1"}, 0) == 0.0
    assert calculate_precision_at_k(["1", "2"], {"1"}, -1) == 0.0

    # Fewer retrieved items than k
    retrieved = ["1", "2"]
    relevant = {"1"}
    # min(5, 2) = 2 -> 1/2 = 0.5
    assert calculate_precision_at_k(retrieved, relevant, 5) == 0.5


def test_calculate_recall_at_k_normal():
    retrieved = ["1", "2", "3", "4", "5"]
    relevant = {"1", "3", "7", "9"}  # 4 total relevant items

    # Top 3: ["1", "2", "3"] -> 2 relevant found out of 4 = 2/4 = 0.5
    assert pytest.approx(calculate_recall_at_k(retrieved, relevant, 3), 0.001) == 0.5

    # Top 5: ["1", "2", "3", "4", "5"] -> 2 relevant found out of 4 = 2/4 = 0.5
    assert pytest.approx(calculate_recall_at_k(retrieved, relevant, 5), 0.001) == 0.5


def test_calculate_recall_at_k_edge_cases():
    assert calculate_recall_at_k([], {"1", "2"}, 5) == 0.0
    assert calculate_recall_at_k(["1", "2"], set(), 5) == 0.0
    assert calculate_recall_at_k(["1", "2"], {"1"}, 0) == 0.0
    assert calculate_recall_at_k(["1", "2"], {"1"}, -1) == 0.0

    # All relevant found in top 2
    retrieved = ["10", "20", "30"]
    relevant = {"10", "20"}
    assert calculate_recall_at_k(retrieved, relevant, 2) == 1.0


def test_extract_candidate_ids():
    class CandidateObj:
        def __init__(self, recipe_id: str):
            self.recipe_id = recipe_id

    candidates = [
        {"recipe_id": "101", "title": "Recipe 1"},
        {"id": "102", "title": "Recipe 2"},
        {"candidateId": "103", "title": "Recipe 3"},
        CandidateObj("104"),
    ]

    extracted = extract_candidate_ids(candidates)
    assert extracted == ["101", "102", "103", "104"]
    assert extract_candidate_ids([]) == []


@pytest.mark.asyncio
async def test_evaluate_retrieval_dataset_empty():
    results = await evaluate_retrieval_dataset([], k_values=[3, 5])
    assert results == {
        "precision@3": 0.0,
        "precision@5": 0.0,
        "recall@3": 0.0,
        "recall@5": 0.0,
    }


@pytest.mark.asyncio
async def test_evaluate_retrieval_dataset_mocked():
    dataset = [
        {
            "id": "test-1",
            "input": {
                "mood": "quick",
                "mealType": "dinner",
                "excludeUrls": ["recipe://bad-1"],
            },
            "relevantRecipeIds": ["1", "2", "3", "4"],
        },
        {
            "id": "test-2",
            "input": {
                "mood": "healthy",
                "mealType": "breakfast",
                "excludeUrls": [],
            },
            "relevantRecipeIds": ["10", "20"],
        },
    ]

    # Mock retrieve function returning different candidates based on mood
    async def mock_retrieve(mood, meal_type, match_count, **kwargs):
        if mood == "quick":
            return [
                {"id": "1", "url": "recipe://good-1"},
                {"id": "bad-1", "url": "recipe://bad-1"},
                {"id": "2", "url": "recipe://good-2"},
                {"id": "99", "url": "recipe://other-99"},
            ]
        else:
            return [
                {"id": "10", "url": "recipe://good-10"},
                {"id": "20", "url": "recipe://good-20"},
                {"id": "30", "url": "recipe://good-30"},
            ]

    results = await evaluate_retrieval_dataset(
        dataset=dataset,
        k_values=[1, 2, 3],
        retrieve_fn=mock_retrieve,
    )

    # Sample 1:
    # Retrieved: ["1", "bad-1", "2", "99"]
    # Filtered excludes "bad-1" -> ["1", "2", "99"]
    # Relevant: {"1", "2", "3", "4"} (4 items)
    # k=1: top_k=["1"] -> p=1.0, r=1/4=0.25
    # k=2: top_k=["1", "2"] -> p=1.0, r=2/4=0.5
    # k=3: top_k=["1", "2", "99"] -> p=2/3, r=2/4=0.5

    # Sample 2:
    # Retrieved & filtered: ["10", "20", "30"]
    # Relevant: {"10", "20"} (2 items)
    # k=1: top_k=["10"] -> p=1.0, r=1/2=0.5
    # k=2: top_k=["10", "20"] -> p=1.0, r=2/2=1.0
    # k=3: top_k=["10", "20", "30"] -> p=2/3, r=2/2=1.0

    # Averages:
    # precision@1 = (1.0 + 1.0) / 2 = 1.0
    # precision@2 = (1.0 + 1.0) / 2 = 1.0
    # precision@3 = (2/3 + 2/3) / 2 = 2/3 (~0.6667)
    # recall@1 = (0.25 + 0.5) / 2 = 0.375
    # recall@2 = (0.5 + 1.0) / 2 = 0.75
    # recall@3 = (0.5 + 1.0) / 2 = 0.75

    assert pytest.approx(results["precision@1"], 0.001) == 1.0
    assert pytest.approx(results["precision@2"], 0.001) == 1.0
    assert pytest.approx(results["precision@3"], 0.001) == 2 / 3
    assert pytest.approx(results["recall@1"], 0.001) == 0.375
    assert pytest.approx(results["recall@2"], 0.001) == 0.75
    assert pytest.approx(results["recall@3"], 0.001) == 0.75


def test_format_evaluation_summary():
    results = {
        "precision@3": 0.85,
        "precision@5": 0.70,
        "recall@3": 0.60,
        "recall@5": 0.90,
    }
    summary = format_evaluation_summary(
        results=results,
        k_values=[3, 5],
        total_samples=30,
        dataset_name="sample.json",
    )

    assert "RAPORT EWALUACJI FAZY RETRIEVALU (RAG)" in summary
    assert "sample.json" in summary
    assert "30" in summary
    assert "Precision@3" in summary
    assert "Recall@5" in summary
    assert "85.00%" in summary
    assert "90.00%" in summary
    assert "OPIS METRYK I WSKAZÓWKI DO OPTYMALIZACJI" in summary


@pytest.mark.asyncio
async def test_cli_main_execution(tmp_path, monkeypatch):
    from scripts.evaluate_retrieval import main

    test_dataset = [
        {
            "id": "sample-1",
            "input": {"mood": "quick", "mealType": "dinner"},
            "relevantRecipeIds": ["1", "2"],
        }
    ]
    dataset_file = tmp_path / "test_data.json"
    dataset_file.write_text(json.dumps(test_dataset), encoding="utf-8")
    output_file = tmp_path / "out.json"

    # Mock evaluate_retrieval_dataset
    mock_results = {"precision@3": 0.5, "recall@3": 1.0}
    mock_eval = AsyncMock(return_value=mock_results)
    monkeypatch.setattr(
        "scripts.evaluate_retrieval.evaluate_retrieval_dataset",
        mock_eval,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_retrieval.py",
            "-d",
            str(dataset_file),
            "-k",
            "3",
            "-o",
            str(output_file),
        ],
    )

    exit_code = await main()
    assert exit_code == 0
    assert output_file.exists()
    saved = json.loads(output_file.read_text(encoding="utf-8"))
    assert saved["metrics"] == mock_results
    assert saved["totalSamples"] == 1
    assert saved["kValues"] == [3]
