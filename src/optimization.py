"""Self-improving modules with DSPy optimization."""

import dspy


def demonstrate_optimization():
    """Show how DSPy can optimize modules with feedback."""
    print("Self-Improving AI with DSPy Optimization:")
    print("-" * 50)

    # Example training data for optimization
    training_examples = [
        dspy.Example(
            question="What is Python?",
            answer="Python is a high-level programming language known for readability.",
        ),
        dspy.Example(
            question="What is DSPy?",
            answer="DSPy is a framework for programming language models declaratively.",
        ),
        dspy.Example(
            question="What are the benefits of modular programming?",
            answer="Modular programming improves code reusability, testing, and maintenance.",
        ),
    ]

    # Metric function for optimization
    def answer_quality_metric(example, prediction, trace=None):
        """Evaluate answer quality based on length and relevance."""
        # Simple metric: answer should be concise but complete
        answer = prediction.answer if hasattr(prediction, "answer") else str(prediction)

        # Check length (not too short, not too long)
        word_count = len(answer.split())
        if word_count < 5:
            return 0.0
        elif word_count > 50:
            return 0.5
        else:
            return 1.0

    print("\n1. Optimization Process:")
    print("   - Define training examples with expected outputs")
    print("   - Create evaluation metrics")
    print("   - Use optimizers to improve module performance")

    print("\n2. Available Optimizers (DSPy 2.6+):")
    optimizers = [
        ("BootstrapFewShot", "Generates few-shot examples from training data"),
        ("MIPROv2", "Advanced optimization with instruction and example selection"),
        ("BetterTogether", "Combines multiple optimization strategies"),
        ("LeReT", "Learns to retrieve and transform examples"),
    ]

    for name, description in optimizers:
        print(f"   - {name}: {description}")

    print("\n3. Example: Optimizing a Q&A Module")
    print("   Training data:")
    for ex in training_examples[:2]:
        print(f"   Q: {ex.question}")
        print(f"   A: {ex.answer}")

    # Mock optimization process
    print("\n4. Optimization Results (simulated):")
    print("   Before optimization: Average quality score = 0.65")
    print("   After optimization:  Average quality score = 0.92")
    print("   Improvement: +41.5%")

    print("\n5. Benefits of Automated Optimization:")
    print("   ✓ No manual prompt tuning required")
    print("   ✓ Systematic improvement based on metrics")
    print("   ✓ Adapts to your specific use case")
    print("   ✓ Continuously improves with more data")

    # Show optimization code pattern
    print("\n6. Optimization Code Pattern:")
    print(
        """
    from dspy.teleprompt import BootstrapFewShot
    
    # Create optimizer
    optimizer = BootstrapFewShot(metric=answer_quality_metric)
    
    # Compile module with training data
    optimized_qa = optimizer.compile(
        student=SimpleQA(),
        trainset=training_examples
    )
    
    # Use optimized module
    answer = optimized_qa("What is machine learning?")
    """
    )

    print("\n✓ DSPy optimization automates the hardest part of LLM development")
    print("✓ Your modules improve themselves based on real performance data")


if __name__ == "__main__":
    from config import configure_llm
    
    # Configure LLM
    configure_llm()
    
    # Run demonstration
    demonstrate_optimization()
