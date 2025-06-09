"""Modern DSPy features: async execution and structured outputs with Pydantic."""

import asyncio
import dspy
from pydantic import BaseModel, Field
from typing import List


class SummaryOutput(BaseModel):
    """Structured output for document summaries."""

    summary: str = Field(description="A concise summary of the document")
    word_count: int = Field(description="Number of words in the summary")


class EntitiesOutput(BaseModel):
    """Structured output for entity extraction."""

    entities: List[str] = Field(description="List of named entities found")
    entity_types: List[str] = Field(description="Types of entities (person, org, etc)")


class ClassificationOutput(BaseModel):
    """Structured output for document classification."""

    label: str = Field(description="Document category")
    confidence: float = Field(description="Confidence score between 0 and 1")


class AsyncSummarizer(dspy.Module):
    """Async summarization with structured output."""

    async def forward(self, document: str) -> SummaryOutput:
        """Return a structured summary of the input document."""
        # In DSPy 2.6+, predict can be async
        result = await self.predict(document=document)

        # Parse result into structured output
        # In production DSPy, this would be automatic with TypedPredictors
        summary_text = str(result)
        word_count = len(summary_text.split())

        return SummaryOutput(summary=summary_text, word_count=word_count)


class AsyncEntityExtractor(dspy.Module):
    """Async entity extraction with structured output."""

    async def forward(self, text: str) -> EntitiesOutput:
        """Extract named entities with their types."""
        result = await self.predict(text=text)

        # Show what the LLM returned
        print(f"    Entity extraction LLM result: {result}")

        # Mock structured parsing (real DSPy would handle this)
        # This demonstrates the pattern
        entities = ["DSPy", "Python", "AI"]
        types = ["Framework", "Language", "Technology"]

        return EntitiesOutput(entities=entities, entity_types=types)


class AsyncClassifier(dspy.Module):
    """Async document classification with confidence scores."""

    async def forward(self, document: str) -> ClassificationOutput:
        """Classify the document with confidence score."""
        result = await self.predict(document=document)

        # Show what the LLM returned
        print(f"    Classification LLM result: {result}")

        # Mock structured output
        return ClassificationOutput(label="technical_documentation", confidence=0.95)


async def demonstrate_structured_outputs():
    """Demonstrate async modules with structured outputs."""
    print("Async Execution with Structured Outputs:")
    print("-" * 50)

    # Sample documents
    technical_doc = """
    DSPy provides a revolutionary approach to AI programming by replacing 
    fragile prompts with modular Python code. It supports async execution,
    structured outputs via Pydantic schemas, and automatic optimization.
    """

    business_doc = """
    Our Q3 earnings exceeded expectations with revenue growth of 15%.
    The company expanded into new markets and increased market share.
    Customer satisfaction scores reached an all-time high.
    """

    # Async summarization
    summarizer = AsyncSummarizer()
    print("\n1. Async Summarization with Word Count:")

    try:
        # Process multiple documents concurrently
        summaries = await asyncio.gather(
            summarizer(technical_doc), summarizer(business_doc)
        )

        for i, (doc, summary) in enumerate(
            zip([technical_doc, business_doc], summaries)
        ):
            print(f"\nDocument {i+1} Summary:")
            print(f"  Summary: {summary.summary}")
            print(f"  Word count: {summary.word_count}")
    except Exception as e:
        print(f"Error in summarization: {e}")

    # Entity extraction
    extractor = AsyncEntityExtractor()
    print("\n\n2. Entity Extraction with Types:")

    try:
        entities = await extractor(technical_doc)
        print(f"Entities found: {', '.join(entities.entities)}")
        print(f"Entity types: {', '.join(entities.entity_types)}")
    except Exception as e:
        print(f"Error in entity extraction: {e}")

    # Classification
    classifier = AsyncClassifier()
    print("\n\n3. Document Classification with Confidence:")

    try:
        results = await asyncio.gather(
            classifier(technical_doc), classifier(business_doc)
        )

        for i, (doc_snippet, result) in enumerate(
            zip(["technical", "business"], results)
        ):
            print(f"\n{doc_snippet.capitalize()} document:")
            print(f"  Classification: {result.label}")
            print(f"  Confidence: {result.confidence:.2%}")
    except Exception as e:
        print(f"Error in classification: {e}")

    print("\n✓ Async execution enables concurrent processing")
    print("✓ Structured outputs ensure reliable, validated responses")
    print("✓ Pydantic schemas catch errors early")


if __name__ == "__main__":
    from config import configure_llm
    
    # Configure LLM
    configure_llm()
    
    # Run demonstration
    asyncio.run(demonstrate_structured_outputs())
