import pytest
from agentscope.tools.ir_evaluator import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k, hit_rate_at_k
)


def test_precision_perfect():
    assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_partial():
    assert precision_at_k(["a", "b", "c"], {"a", "b"}, 3) == pytest.approx(2 / 3)


def test_precision_none():
    assert precision_at_k(["x", "y", "z"], {"a", "b", "c"}, 3) == 0.0


def test_recall_perfect():
    assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0


def test_recall_partial():
    assert recall_at_k(["a", "b", "c"], {"a", "b", "d"}, 3) == pytest.approx(2 / 3)


def test_recall_empty_relevant():
    assert recall_at_k(["a", "b"], set(), 2) == 0.0


def test_mrr_first():
    assert mrr([["a", "b"]], [{"a"}]) == 1.0


def test_mrr_second():
    assert mrr([["b", "a"]], [{"a"}]) == pytest.approx(0.5)


def test_mrr_multiple():
    assert mrr([["a", "b"], ["c", "a"]], [{"a"}, {"a"}]) == pytest.approx(0.75)


def test_mrr_no_hit():
    assert mrr([["x", "y"]], [{"a"}]) == 0.0


def test_ndcg_perfect():
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_zero():
    assert ndcg_at_k(["x", "y"], {"a", "b"}, 2) == pytest.approx(0.0)


def test_ndcg_partial():
    score = ndcg_at_k(["a", "x"], {"a", "b"}, 2)
    assert 0.0 < score < 1.0


def test_hit_rate_all_hit():
    assert hit_rate_at_k([["a", "b"], ["c", "d"]], [{"a"}, {"c"}], 2) == 1.0


def test_hit_rate_half():
    assert hit_rate_at_k([["a"], ["x"]], [{"a"}, {"z"}], 1) == 0.5


def test_hit_rate_none():
    assert hit_rate_at_k([["x"], ["y"]], [{"a"}, {"b"}], 1) == 0.0
