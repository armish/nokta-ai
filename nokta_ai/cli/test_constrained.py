#!/usr/bin/env python3
"""
Quick test script for constrained diacritics restoration model
"""

import argparse
from ..models.constrained import ConstrainedDiacriticsRestorer, remove_diacritics_simple


def test_constrained_model(model_path: str, context_size: int = None):
    """Quick interactive test of constrained model"""
    print(f"Loading constrained model from {model_path}...")
    if context_size:
        print(f"Using specified context size: {context_size}")
        restorer = ConstrainedDiacriticsRestorer(model_path=model_path, context_size=context_size)
    else:
        print("Using context size from model checkpoint")
        restorer = ConstrainedDiacriticsRestorer(model_path=model_path)

    print("\nConstrained Turkish Diacritics Restoration - Test Mode")
    print("=" * 55)
    print("Enter Turkish text (or 'quit' to exit)")
    print("The model will restore diacritics for: c/ç, g/ğ, i/ı, o/ö, s/ş, u/ü")
    print()

    # Predefined test cases
    test_cases = [
        "Bugun hava cok guzel",
        "Ogrenciler sinifta ders calisiyor",
        "Turkiye'nin baskenti Ankara'dir",
        "Cocuklar bahcede futbol oynuyorlar",
        "Universiteye gitmek icin erken kalktim"
    ]

    print("Running predefined test cases:")
    print("-" * 40)

    for i, test_text in enumerate(test_cases, 1):
        restored = restorer.restore_diacritics(test_text)
        print(f"{i}. Input:    {test_text}")
        print(f"   Output:   {restored}")
        print()

    print("Interactive mode (Ctrl+C to exit):")
    print("-" * 40)

    try:
        while True:
            text = input("Input: ").strip()

            if text.lower() in ['quit', 'exit', 'q']:
                break

            if not text:
                continue

            # Show original without diacritics
            stripped = remove_diacritics_simple(text)
            if stripped != text:
                print(f"Stripped: {stripped}")

            # Restore diacritics
            restored = restorer.restore_diacritics(stripped)
            print(f"Output:   {restored}")

            # Show length preservation
            print(f"Lengths:  {len(text)} → {len(restored)} (preserved: {len(text) == len(restored)})")
            print()

    except KeyboardInterrupt:
        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Test constrained diacritics model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained constrained model')
    parser.add_argument('--context-size', type=int,
                       help='Override context size (use model default if not specified)')

    args = parser.parse_args()
    test_constrained_model(args.model, args.context_size)


if __name__ == "__main__":
    main()