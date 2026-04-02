import ast
import pytest


def test_synth_gen_parses():
    """synth_gen.py must parse without syntax errors."""
    with open("agentscope/tools/synth_gen.py") as f:
        source = f.read()
    ast.parse(source)


def test_synth_gen_has_run_function():
    with open("agentscope/tools/synth_gen.py") as f:
        source = f.read()
    assert "def run(state" in source


def test_synth_gen_uses_synthesizer():
    with open("agentscope/tools/synth_gen.py") as f:
        source = f.read()
    assert "Synthesizer" in source


def test_synth_gen_enforces_cap():
    with open("agentscope/tools/synth_gen.py") as f:
        source = f.read()
    assert "max_synth_pairs" in source
    assert "goldens[:max_pairs]" in source


def test_synth_gen_writes_gt_path():
    with open("agentscope/tools/synth_gen.py") as f:
        source = f.read()
    assert 'state["gt_path"]' in source


def test_synth_gen_writes_synth_results():
    with open("agentscope/tools/synth_gen.py") as f:
        source = f.read()
    assert 'state["synth_results"]' in source
    assert "coverage_score" in source
