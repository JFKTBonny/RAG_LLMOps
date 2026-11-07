# rag_08_full_pipeline.py — Complete production RAG pipeline

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_project.rag_pipeline import TechCorpRAGPipeline

print("═" * 60)
print("  TECHCORP RAG ASSISTANT — Powered by Local AI")
print("═" * 60)

pipeline = TechCorpRAGPipeline()
print("\nKnowledge base ready.")
print("\nReady! Ask me anything about the TechCorp handbook.")
print("Type 'quit' to exit, 'debug' before your question to see retrieved chunks.\n")

while True:
    user_input = input("You: ").strip()

    if not user_input:
        continue

    EXIT_COMMANDS = ["quit", "exit", "bye", "goodbye", "that is all", "thanks"]
    if any(cmd in user_input.lower() for cmd in EXIT_COMMANDS):
        print("Goodbye!")
        break

    debug_mode = False
    if user_input.lower().startswith("debug "):
        debug_mode = True
        user_input = user_input[6:]

    print("Thinking...")
    answer, faith_score, chunks_used = pipeline.ask(user_input)

    print(f"\nAssistant: {answer}")
    print(f"[Faithfulness: {faith_score:.2f}]")

    if debug_mode:
        print("\nRetrieved chunks:")
        for i, chunk in enumerate(chunks_used, 1):
            print(f"  {i}. {chunk[:100]}...")

    print()
