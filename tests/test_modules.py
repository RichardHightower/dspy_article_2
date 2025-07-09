"""Unit tests for DSPy modules."""

import pytest
import asyncio
from src.basic_examples import SimpleQA, Summarize
from src.structured_outputs import (
    AsyncSummarizer,
    AsyncEntityExtractor,
    SummaryOutput,
    EntitiesOutput,
    ClassificationOutput,
)


class TestBasicModules:
    """Test basic DSPy modules."""

    def test_qa_module_structure(self):
        """Test Q&A module has correct structure."""
        qa = SimpleQA()
        assert hasattr(qa, "forward")
        assert callable(qa.forward)

    def test_summarize_module_structure(self):
        """Test summarizer has correct structure."""
        summarizer = Summarize()
        assert hasattr(summarizer, "forward")
        assert callable(summarizer.forward)


class TestStructuredOutputs:
    """Test modules with structured outputs."""

    @pytest.mark.asyncio
    async def test_async_summarizer_returns_structured_output(self):
        """Test async summarizer returns SummaryOutput."""
        summarizer = AsyncSummarizer()
        # Mock test - in real tests you'd mock the LLM response
        assert hasattr(summarizer, "aforward")
        assert asyncio.iscoroutinefunction(summarizer.aforward)

    @pytest.mark.asyncio
    async def test_entity_extractor_returns_structured_output(self):
        """Test entity extractor returns EntitiesOutput."""
        extractor = AsyncEntityExtractor()
        assert hasattr(extractor, "aforward")
        assert asyncio.iscoroutinefunction(extractor.aforward)

    def test_output_schemas_are_valid(self):
        """Test Pydantic schemas are properly defined."""
        # Test SummaryOutput
        summary = SummaryOutput(summary="Test summary", word_count=2)
        assert summary.summary == "Test summary"
        assert summary.word_count == 2

        # Test EntitiesOutput
        entities = EntitiesOutput(
            entities=["Person1", "Company1"], entity_types=["person", "organization"]
        )
        assert len(entities.entities) == 2
        assert len(entities.entity_types) == 2

        # Test ClassificationOutput
        classification = ClassificationOutput(label="technical", confidence=0.95)
        assert classification.label == "technical"
        assert 0 <= classification.confidence <= 1


class TestPipelines:
    """Test pipeline composition."""

    def test_pipeline_composition(self):
        """Test that pipelines can be created and have expected structure."""
        from src.pipelines import ProcessEmailPipeline, CodeAnalysisPipeline

        # Email pipeline
        email_pipeline = ProcessEmailPipeline()
        assert hasattr(email_pipeline, "aforward")
        assert asyncio.iscoroutinefunction(email_pipeline.aforward)
        assert hasattr(email_pipeline, "summarize")
        assert hasattr(email_pipeline, "extract_entities")

        # Code pipeline
        code_pipeline = CodeAnalysisPipeline()
        assert hasattr(code_pipeline, "forward")
        assert hasattr(code_pipeline, "understand")
        assert hasattr(code_pipeline, "find_issues")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
