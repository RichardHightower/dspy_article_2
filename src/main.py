"""Main entry point demonstrating all DSPy examples from the chapter."""

import asyncio
from src.config import configure_llm
from src.basic_examples import demonstrate_basic_qa, demonstrate_fragile_prompts
from src.structured_outputs import demonstrate_structured_outputs
from src.pipelines import demonstrate_pipeline
from src.optimization import demonstrate_optimization


def print_header(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70 + "\n")


async def main():
    """Run all chapter examples."""
    print_header("DSPy: Beyond Prompt Hacking")
    print("Moving from fragile prompts to robust, modular AI programming")

    # Configure LLM
    configure_llm()

    # Demonstrate the problems with prompt engineering
    print_header("The Problem: Fragile Prompt Engineering")
    demonstrate_fragile_prompts()

    # Show basic DSPy modules
    print_header("The Solution: DSPy Modules")
    demonstrate_basic_qa()

    # Demonstrate structured outputs with async
    print_header("Modern DSPy: Async & Structured Outputs")
    await demonstrate_structured_outputs()

    # Show pipeline composition
    print_header("Composing Modules into Pipelines")
    await demonstrate_pipeline()

    # Demonstrate optimization
    print_header("Self-Improving AI with Optimization")
    demonstrate_optimization()

    print_header("Summary")
    print("✓ DSPy replaces fragile prompts with modular, testable code")
    print("✓ Async support and structured outputs make it production-ready")
    print("✓ Modules compose into powerful pipelines")
    print("✓ Automatic optimization improves performance without manual tuning")
    print("\nExplore individual modules in src/ to learn more!")


if __name__ == "__main__":
    asyncio.run(main())
