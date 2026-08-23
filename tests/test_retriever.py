"""
Tests for UniversalDocumentRetriever keyword search and ranking in agent.py.
"""

import pytest
from pathlib import Path
from agent import UniversalDocumentRetriever


@pytest.fixture
def sample_corpus(tmp_path):
    """Creates a temporary multi-topic corpus."""
    doc1 = tmp_path / "ai_safety.md"
    doc1.write_text(
        "## RLAIF Overview\nReinforcement Learning from AI Feedback achieved 93.4% safety compliance.\n"
        "## SAEs\nSparse Autoencoders isolate monosemantic features.",
        encoding="utf-8"
    )
    doc2 = tmp_path / "batteries.md"
    doc2.write_text(
        "## Iron-Air\nIron-air batteries cost approximately $20/kWh for 100-hour discharge.\n"
        "## Vanadium Flow\nVanadium redox flow batteries have zero degradation over 25000 cycles.",
        encoding="utf-8"
    )
    return tmp_path


def test_retriever_keyword_matching(sample_corpus):
    """Verifies that relevant passages are retrieved with correct metadata."""
    retriever = UniversalDocumentRetriever(data_path=sample_corpus)
    
    # Query for RLAIF
    results = retriever.search("How does RLAIF perform in safety compliance?", top_k=2)
    assert len(results) > 0
    assert results[0]["section"] == "RLAIF Overview"
    assert "93.4%" in results[0]["content"]


def test_retriever_out_of_domain(sample_corpus):
    """Verifies that queries with no matching domain terms return empty list."""
    retriever = UniversalDocumentRetriever(data_path=sample_corpus)
    
    # Query unrelated to the sample corpus
    results = retriever.search("What are the primary rovers currently operating on the surface of Mars?", top_k=3)
    assert results == []
