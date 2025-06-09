"""Basic DSPy examples showing the move from prompts to modules."""

import dspy


def demonstrate_fragile_prompts():
    """Show how small prompt changes can cause different behaviors."""
    print("Traditional Prompt Engineering Problems:")
    print("-" * 50)

    # Example of fragile prompts
    prompt1 = "Summarize the following document:"
    prompt2 = "Please provide a summary of this document:"

    print(f"Prompt 1: {prompt1}")
    print(f"Prompt 2: {prompt2}")
    print("\nThese nearly identical prompts can produce different results!")
    print("- Different output lengths")
    print("- Different styles (formal vs informal)")
    print("- Unpredictable failures with model updates")
    print("\nThis fragility makes prompt-based systems hard to maintain.\n")


class SimpleQA(dspy.Module):
    """A basic question-answering module."""

    def forward(self, question: str) -> str:
        """Answer a factual question concisely."""
        return self.predict(question=question)


class Summarize(dspy.Module):
    """A basic summarization module."""

    def forward(self, document: str) -> str:
        """Summarize the input document in 2-3 sentences."""
        return self.predict(document=document)


def demonstrate_basic_qa():
    """Demonstrate basic DSPy modules for Q&A and summarization."""
    print("DSPy Solution: Modular, Code-Driven AI")
    print("-" * 50)

    # Question Answering
    qa = SimpleQA()
    questions = [
        "What is DSPy?",
        "What are the benefits of modular AI programming?",
        "How does DSPy differ from prompt engineering?",
    ]

    print("Question Answering Module:")
    for question in questions:
        try:
            answer = qa(question)
            print(f"\nQ: {question}")
            print(f"A: {answer}")
        except Exception as e:
            print(f"\nError answering '{question}': {e}")

    # Summarization
    print("\n\nSummarization Module:")
    summarizer = Summarize()

    sample_text = """
    DSPy is a framework for programming language models that brings software 
    engineering best practices to AI development. Instead of writing fragile 
    prompts, developers define modules with clear inputs and outputs. DSPy 
    automatically generates and optimizes the prompts behind the scenes, 
    making AI systems more reliable and maintainable.
    """

    try:
        summary = summarizer(sample_text)
        print(f"\nOriginal text: {sample_text.strip()}")
        print(f"\nSummary: {summary}")
    except Exception as e:
        print(f"Error summarizing: {e}")

    print("\n✓ Notice how we never wrote prompts - just defined what we wanted!")


if __name__ == "__main__":
    from config import configure_llm
    
    # Configure LLM
    configure_llm()
    
    # Run demonstrations
    demonstrate_fragile_prompts()
    demonstrate_basic_qa()
