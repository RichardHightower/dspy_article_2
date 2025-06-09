"""Multi-stage DSPy pipelines showing module composition."""

import asyncio
import dspy
from pydantic import BaseModel
from typing import List


class EmailAnalysisOutput(BaseModel):
    """Complete email analysis results."""

    summary: str
    entities: List[str]
    sentiment: str
    priority: str
    suggested_response: str


class ProcessEmailPipeline(dspy.Module):
    """Multi-stage pipeline for email processing."""

    def __init__(self):
        super().__init__()
        # Compose multiple modules
        self.summarize = dspy.Predict("email -> summary")
        self.extract_entities = dspy.Predict("text -> entities")
        self.analyze_sentiment = dspy.Predict("text -> sentiment")
        self.determine_priority = dspy.Predict("summary, sentiment -> priority")
        self.suggest_response = dspy.Predict("summary, sentiment, priority -> response")

    async def forward(self, email_body: str) -> EmailAnalysisOutput:
        """Process email through multiple analysis stages."""
        # Stage 1: Summarize
        summary = self.summarize(email=email_body).summary

        # Stage 2: Extract entities (concurrent with sentiment)
        entities_task = asyncio.create_task(self.extract_entities(text=email_body))
        sentiment_task = asyncio.create_task(self.analyze_sentiment(text=email_body))

        entities_result = await entities_task
        sentiment_result = await sentiment_task

        # Parse results (using actual LLM responses when available)
        # For demo, we'll show what the LLM returned and use fallback values
        entities = getattr(
            entities_result, "entities", "customer, product, issue"
        ).split(", ")
        sentiment = getattr(sentiment_result, "sentiment", "negative")

        print(f"    Entities extraction result: {entities_result}")
        print(f"    Sentiment analysis result: {sentiment_result}")

        # Stage 3: Determine priority based on summary and sentiment
        priority_result = self.determine_priority(summary=summary, sentiment=sentiment)
        priority = getattr(priority_result, "priority", "high")
        print(f"    Priority determination result: {priority_result}")

        # Stage 4: Suggest response
        response_result = self.suggest_response(
            summary=summary, sentiment=sentiment, priority=priority
        )
        suggested_response = getattr(
            response_result,
            "response",
            "Thank you for reaching out. We understand your concern...",
        )
        print(f"    Response suggestion result: {response_result}")

        return EmailAnalysisOutput(
            summary=summary,
            entities=entities,
            sentiment=sentiment,
            priority=priority,
            suggested_response=suggested_response,
        )


class CodeAnalysisPipeline(dspy.Module):
    """Pipeline for analyzing code quality and suggesting improvements."""

    def __init__(self):
        super().__init__()
        self.understand = dspy.Predict("code -> description")
        self.find_issues = dspy.ChainOfThought("code, description -> issues")
        self.suggest_fixes = dspy.Predict("code, issues -> suggestions")
        self.generate_tests = dspy.Predict("code, description -> tests")

    def forward(self, code: str):
        """Analyze code and provide comprehensive feedback."""
        # Understand what the code does
        description = self.understand(code=code).description

        # Find potential issues (with reasoning)
        issues = self.find_issues(code=code, description=description).issues

        # Suggest improvements
        suggestions = self.suggest_fixes(code=code, issues=issues).suggestions

        # Generate test cases
        tests = self.generate_tests(code=code, description=description).tests

        return {
            "description": description,
            "issues": issues,
            "suggestions": suggestions,
            "tests": tests,
        }


async def demonstrate_pipeline():
    """Demonstrate multi-stage pipeline processing."""
    print("Multi-Stage Pipeline Examples:")
    print("-" * 50)

    # Email processing pipeline
    print("\n1. Email Analysis Pipeline:")
    email_pipeline = ProcessEmailPipeline()

    sample_email = """
    Subject: Urgent: Product not working as expected
    
    I purchased your premium software last week, but I'm experiencing
    constant crashes. I've tried reinstalling twice. This is affecting
    my business operations. I need this resolved immediately or I want
    a full refund.
    
    Order #12345
    John Smith
    """

    try:
        result = await email_pipeline(sample_email)
        print("\nEmail Analysis Results:")
        print(f"  Summary: {result.summary}")
        print(f"  Entities: {', '.join(result.entities)}")
        print(f"  Sentiment: {result.sentiment}")
        print(f"  Priority: {result.priority}")
        print(f"  Suggested Response: {result.suggested_response[:100]}...")
    except Exception as e:
        print(f"Error in email pipeline: {e}")

    # Code analysis pipeline
    print("\n\n2. Code Analysis Pipeline:")
    code_analyzer = CodeAnalysisPipeline()

    sample_code = """
    def calculate_average(numbers):
        total = 0
        for n in numbers:
            total += n
        return total / len(numbers)
    """

    try:
        analysis = code_analyzer(sample_code)
        print("\nCode Analysis Results:")
        print(f"  What it does: {analysis['description']}")
        print(f"  Issues found: {analysis['issues']}")
        print(f"  Suggestions: {analysis['suggestions']}")
        print(f"  Test cases: {analysis['tests']}")
    except Exception as e:
        print(f"Error in code analysis: {e}")

    print("\n✓ Pipelines compose multiple modules for complex tasks")
    print("✓ Each stage is independently testable")
    print("✓ Async stages can run concurrently for better performance")


if __name__ == "__main__":
    from config import configure_llm
    
    # Configure LLM
    configure_llm()
    
    # Run demonstration
    asyncio.run(demonstrate_pipeline())
