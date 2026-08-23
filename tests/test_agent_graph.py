"""
Unit and integration tests for LangGraph StateGraph agent execution, routing, and guardrails.
"""

import pytest
from agent import (
    ResearchAgent,
    route_after_local_check,
    ResearchState,
    retrieve_local_node,
    tavily_search_node,
    generate_answer_node,
    UniversalDocumentRetriever
)


def test_agent_graph_structure():
    """Verifies that ResearchAgent compiles the LangGraph StateGraph with all required nodes."""
    agent = ResearchAgent()
    assert agent.graph is not None
    # Verify graph node names exist
    node_names = list(agent.graph.nodes.keys())
    assert "retrieve_local" in node_names
    assert "tavily_search" in node_names
    assert "generate_answer" in node_names


def test_route_after_local_check_branches():
    """Verifies conditional routing decisions based on local data relevance."""
    # Branch 1: Local data is relevant -> goes directly to answer synthesis
    state_relevant: ResearchState = {
        "question": "How does RLAIF work?",
        "local_docs": [{"file": "ai_safety.md", "section": "RLAIF", "content": "RLAIF achieved 93.4%"}],
        "is_local_relevant": True,
        "web_results": [],
        "retrieval_source": "local",
        "answer": "",
        "citations": []
    }
    assert route_after_local_check(state_relevant) == "generate_answer"

    # Branch 2: Local data is NOT relevant -> falls back to Tavily web search
    state_not_relevant: ResearchState = {
        "question": "What are the latest Mars rover discoveries?",
        "local_docs": [],
        "is_local_relevant": False,
        "web_results": [],
        "retrieval_source": "none",
        "answer": "",
        "citations": []
    }
    assert route_after_local_check(state_not_relevant) == "tavily_search"


def test_retrieve_local_node(tmp_path):
    """Tests the local retrieval node functionality."""
    doc = tmp_path / "test.md"
    doc.write_text("## Quantum\nSurface codes suppress errors.", encoding="utf-8")
    retriever = UniversalDocumentRetriever(data_path=tmp_path)
    
    state: ResearchState = {
        "question": "How do surface codes suppress errors?",
        "local_docs": [],
        "is_local_relevant": False,
        "web_results": [],
        "retrieval_source": "none",
        "answer": "",
        "citations": []
    }
    
    result = retrieve_local_node(state, retriever)
    assert result["is_local_relevant"] is True
    assert len(result["local_docs"]) == 1
    assert result["local_docs"][0]["section"] == "Quantum"


def test_generate_answer_node_fallback():
    """Tests that missing info triggers the guardrail fallback message."""
    empty_state: ResearchState = {
        "question": "Unknown fictional query",
        "local_docs": [],
        "is_local_relevant": False,
        "web_results": [],
        "retrieval_source": "none",
        "answer": "",
        "citations": []
    }
    result = generate_answer_node(empty_state)
    assert result["retrieval_source"] == "none"
    assert len(result["citations"]) == 0
    assert "insufficient" in result["answer"].lower() or "do not contain" in result["answer"].lower()


def test_tavily_search_guardrail_filtering():
    """Tests that fictional/impossible questions are blocked before making search calls."""
    state: ResearchState = {
        "question": "What is the secret alchemy formula for converting Vibranium into dark matter?",
        "local_docs": [],
        "is_local_relevant": False,
        "web_results": [],
        "retrieval_source": "none",
        "answer": "",
        "citations": []
    }
    result = tavily_search_node(state)
    assert result["web_results"] == []
    assert result["retrieval_source"] == "none"
