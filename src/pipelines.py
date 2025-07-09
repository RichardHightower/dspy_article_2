"""Multi-stage DSPy pipelines showing module composition.

Follows DSPy async best practices from ASYNC.md:
- Uses asyncio.gather() for concurrent execution
- Uses native .acall() methods when available
- Properly named async methods (aforward)

Note: This code assumes DSPy 2.6+ with native async support.
For older versions, wrap sync calls with asyncio.to_thread().
"""

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


class CodeAnalysisOutput(BaseModel):
    """Structured output for code analysis results."""

    description: str
    issues: str
    suggestions: str
    tests: str


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

    async def aforward(self, email_body: str) -> EmailAnalysisOutput:
        """Process email through multiple analysis stages."""
        # Stage 1: Summarize (sequential as needed for next stages)
        summary_result = await self.summarize.acall(email=email_body)
        summary = summary_result.summary

        # Stage 2: Extract entities and analyze sentiment concurrently
        # Using asyncio.gather() as recommended by DSPy documentation
        entities_result, sentiment_result = await asyncio.gather(
            self.extract_entities.acall(text=email_body),
            self.analyze_sentiment.acall(text=email_body),
            return_exceptions=True  # Handle partial failures gracefully
        )

        # Handle potential errors and parse results
        if isinstance(entities_result, Exception):
            print(f"    Error during entity extraction: {entities_result}")
            entities = ["customer", "product", "issue"]  # Fallback values
        else:
            entities = getattr(
                entities_result, "entities", "customer, product, issue"
            ).split(", ")
            print(f"    Entities extraction result: {entities_result}")
        
        if isinstance(sentiment_result, Exception):
            print(f"    Error during sentiment analysis: {sentiment_result}")
            sentiment = "negative"  # Fallback value
        else:
            sentiment = getattr(sentiment_result, "sentiment", "negative")
            print(f"    Sentiment analysis result: {sentiment_result}")

        # Stage 3: Determine priority based on summary and sentiment
        priority_result = await self.determine_priority.acall(
            summary=summary, sentiment=sentiment
        )
        priority = getattr(priority_result, "priority", "high")
        print(f"    Priority determination result: {priority_result}")

        # Stage 4: Suggest response
        response_result = await self.suggest_response.acall(
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

    def forward(self, code: str) -> CodeAnalysisOutput:
        """Analyze code and provide comprehensive feedback."""
        # Understand what the code does
        description = self.understand(code=code).description

        # Find potential issues (with reasoning)
        issues = self.find_issues(code=code, description=description).issues

        # Suggest improvements
        suggestions = self.suggest_fixes(code=code, issues=issues).suggestions

        # Generate test cases
        tests = self.generate_tests(code=code, description=description).tests

        return CodeAnalysisOutput(
            description=description,
            issues=issues,
            suggestions=suggestions,
            tests=tests,
        )

    async def aforward(self, code: str) -> CodeAnalysisOutput:
        """Analyze code asynchronously with concurrent stage execution where possible."""
        # Stage 1: Understand what the code does
        description_result = await self.understand.acall(code=code)
        description = description_result.description

        # Stage 2: Find issues and generate tests concurrently
        # Both can run in parallel since they only depend on code and description
        issues_task = self.find_issues.acall(code=code, description=description)
        tests_task = self.generate_tests.acall(code=code, description=description)
        
        issues_result, tests_result = await asyncio.gather(
            issues_task, 
            tests_task,
            return_exceptions=True  # Handle partial failures gracefully
        )
        
        # Handle potential errors
        issues = issues_result.issues if not isinstance(issues_result, Exception) else "Error finding issues."
        tests = tests_result.tests if not isinstance(tests_result, Exception) else "Error generating tests."
        
        # Log actual exceptions for debugging
        if isinstance(issues_result, Exception):
            print(f"Error during issue finding: {issues_result}")
        if isinstance(tests_result, Exception):
            print(f"Error during test generation: {tests_result}")

        # Stage 3: Suggest fixes based on issues
        suggestions_result = await self.suggest_fixes.acall(code=code, issues=issues)
        suggestions = suggestions_result.suggestions

        return CodeAnalysisOutput(
            description=description,
            issues=issues,
            suggestions=suggestions,
            tests=tests,
        )


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
        # Note: we call aforward() for async execution
        result = await email_pipeline.aforward(sample_email)
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
        print(f"  What it does: {analysis.description}")
        print(f"  Issues found: {analysis.issues}")
        print(f"  Suggestions: {analysis.suggestions}")
        print(f"  Test cases: {analysis.tests}")
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
