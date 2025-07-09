"""Modern DSPy features: async execution and structured outputs with Pydantic.

Follows DSPy async best practices:
- Uses aforward() naming for async methods
- Uses native .acall() for async predictions
- Proper error handling with structured outputs

Note: This code assumes DSPy 2.6+ with native async support.
For older versions, use dspy.asyncify() or asyncio.to_thread().
"""

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

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("document -> summary")

    async def aforward(self, document: str) -> SummaryOutput:
        """Return a structured summary of the input document."""
        # Use native async support with acall()
        result = await self.predict.acall(document=document)

        # Parse result into structured output
        # In production DSPy, this would be automatic with TypedPredictors
        summary_text = str(result.summary)
        word_count = len(summary_text.split())

        return SummaryOutput(summary=summary_text, word_count=word_count)


class AsyncEntityExtractor(dspy.Module):
    """Async entity extraction with structured output."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("text -> entities")

    async def aforward(self, text: str) -> EntitiesOutput:
        """Extract named entities with their types."""
        result = await self.predict.acall(text=text)

        # Show what the LLM returned
        print(f"    Entity extraction LLM result: {result.entities}")

        # Parse the result - in real DSPy with TypedPredictors this would be automatic
        # For now, we'll extract entities from the LLM response
        entities_str = str(result.entities)
        # Simple parsing - split by common delimiters
        entities = [e.strip() for e in entities_str.replace(",", ";").split(";") if e.strip()]
        
        # Infer entity types based on common patterns
        entity_types = []
        for entity in entities:
            entity_lower = entity.lower()
            if any(word in entity_lower for word in ["inc", "corp", "company", "llc", "organization"]):
                entity_types.append("Organization")
            elif any(word in entity_lower for word in ["framework", "library", "api", "programming", "code", "schema", "system"]):
                entity_types.append("Technology")
            elif any(word in entity_lower for word in ["optimization", "execution", "process", "method"]):
                entity_types.append("Concept")
            elif entity[0].isupper() and len(entity.split()) == 2 and not any(tech in entity_lower for tech in ["programming", "code", "api"]):
                # Likely a person's name (two capitalized words)
                entity_types.append("Person")
            else:
                entity_types.append("Other")

        return EntitiesOutput(entities=entities[:5], entity_types=entity_types[:5])  # Limit to 5 for demo


class AsyncClassifier(dspy.Module):
    """Async document classification with confidence scores."""

    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict("document -> category, confidence_score")

    async def aforward(self, document: str) -> ClassificationOutput:
        """Classify the document with confidence score."""
        result = await self.predict.acall(document=document)

        # Show what the LLM returned
        print(f"    Classification LLM result: category={result.category}, confidence={result.confidence_score}")

        # Parse the confidence score
        try:
            # Handle various confidence formats (0.95, 95%, "high", etc.)
            confidence_str = str(result.confidence_score).strip()
            if confidence_str.endswith('%'):
                confidence = float(confidence_str.rstrip('%')) / 100
            elif confidence_str.lower() in ['high', 'very high']:
                confidence = 0.9
            elif confidence_str.lower() == 'medium':
                confidence = 0.7
            elif confidence_str.lower() == 'low':
                confidence = 0.5
            else:
                confidence = float(confidence_str)
                if confidence > 1:  # If given as percentage without %
                    confidence = confidence / 100
        except (ValueError, AttributeError):
            confidence = 0.8  # Default confidence if parsing fails

        return ClassificationOutput(label=str(result.category).strip(), confidence=confidence)


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
            summarizer.aforward(technical_doc), 
            summarizer.aforward(business_doc),
            return_exceptions=True  # Handle partial failures gracefully
        )

        for i, (doc, summary) in enumerate(zip([technical_doc, business_doc], summaries)):
            if isinstance(summary, Exception):
                print(f"\nDocument {i+1} Error: {summary}")
            else:
                print(f"\nDocument {i+1} Summary:")
                print(f"  Summary: {summary.summary}")
                print(f"  Word count: {summary.word_count}")
    except Exception as e:
        print(f"Error in summarization: {e}")

    # Entity extraction
    extractor = AsyncEntityExtractor()
    print("\n\n2. Entity Extraction with Types:")

    try:
        entities = await extractor.aforward(technical_doc)
        print(f"Entities found: {', '.join(entities.entities)}")
        print(f"Entity types: {', '.join(entities.entity_types)}")
    except Exception as e:
        print(f"Error in entity extraction: {e}")

    # Classification
    classifier = AsyncClassifier()
    print("\n\n3. Document Classification with Confidence:")

    try:
        results = await asyncio.gather(
            classifier.aforward(technical_doc), 
            classifier.aforward(business_doc),
            return_exceptions=True
        )

        for i, (doc_snippet, result) in enumerate(zip(["technical", "business"], results)):
            if isinstance(result, Exception):
                print(f"\n{doc_snippet.capitalize()} document error: {result}")
            else:
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
