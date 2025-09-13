#!/usr/bin/env python3
"""
Inference script for Turkish diacritics restoration
Use this to restore diacritics in any Turkish text
"""

import sys
import argparse
from pathlib import Path
from diacritics_restoration import DiacriticsRestorer, TurkishDiacriticsMapper


def restore_text_file(input_file: str, output_file: str, model_path: str):
    """Restore diacritics in a text file"""
    # Load model
    restorer = DiacriticsRestorer(model_path=model_path)
    mapper = TurkishDiacriticsMapper()

    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Processing {len(text)} characters...")

    # Process line by line for better results
    lines = text.split('\n')
    restored_lines = []

    for i, line in enumerate(lines):
        if line.strip():
            # Remove existing diacritics first (in case of partial diacritics)
            stripped = mapper.remove_diacritics(line)
            # Restore diacritics
            restored = restorer.restore_diacritics(stripped)
            restored_lines.append(restored)
        else:
            restored_lines.append(line)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(lines)} lines")

    # Save output
    restored_text = '\n'.join(restored_lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(restored_text)

    print(f"Restored text saved to {output_file}")


def interactive_mode(model_path: str):
    """Interactive mode for testing diacritics restoration"""
    # Load model
    print(f"Loading model from {model_path}...")
    restorer = DiacriticsRestorer(model_path=model_path)
    mapper = TurkishDiacriticsMapper()

    print("\nTurkish Diacritics Restoration - Interactive Mode")
    print("=" * 50)
    print("Enter Turkish text without diacritics (or 'quit' to exit)")
    print("The model will restore the diacritics.\n")

    while True:
        # Get input
        text = input("Input text: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            break

        if not text:
            continue

        # Remove any existing diacritics (for testing)
        stripped = mapper.remove_diacritics(text)
        print(f"Stripped:  {stripped}")

        # Restore diacritics
        restored = restorer.restore_diacritics(stripped)
        print(f"Restored:  {restored}")
        print()


def benchmark_model(model_path: str):
    """Benchmark model performance on test sentences"""
    # Load model
    restorer = DiacriticsRestorer(model_path=model_path)
    mapper = TurkishDiacriticsMapper()

    # Test sentences with known correct diacritics
    test_sentences = [
        ("Turkiye'nin baskenti Ankara'dir.", "Türkiye'nin başkenti Ankara'dır."),
        ("Ogrenciler sinifta ders calisiyor.", "Öğrenciler sınıfta ders çalışıyor."),
        ("Bugun hava cok guzel ve gunesliydi.", "Bugün hava çok güzel ve güneşliydi."),
        ("Universiteye gitmek icin erken kalktim.", "Üniversiteye gitmek için erken kalktım."),
        ("Cocuklar bahcede futbol oynuyorlar.", "Çocuklar bahçede futbol oynuyorlar."),
        ("Istanbul Bogazı cok guzel gorunuyor.", "İstanbul Boğazı çok güzel görünüyor."),
        ("Aksam yemeginde balik yedik.", "Akşam yemeğinde balık yedik."),
        ("Kitap okumak cok onemlidir.", "Kitap okumak çok önemlidir."),
        ("Muzik dinlemeyi cok seviyorum.", "Müzik dinlemeyi çok seviyorum."),
        ("Yarin hava yagmurlu olacakmis.", "Yarın hava yağmurlu olacakmış.")
    ]

    print("\nModel Benchmark")
    print("=" * 60)

    total_chars = 0
    correct_chars = 0

    for stripped, expected in test_sentences:
        # Ensure input is without diacritics
        input_text = mapper.remove_diacritics(stripped)

        # Restore
        restored = restorer.restore_diacritics(input_text)

        # Calculate accuracy
        chars_correct = sum(1 for c1, c2 in zip(expected, restored) if c1 == c2)
        accuracy = chars_correct / len(expected) * 100

        total_chars += len(expected)
        correct_chars += chars_correct

        print(f"\nInput:    {input_text}")
        print(f"Expected: {expected}")
        print(f"Restored: {restored}")
        print(f"Accuracy: {accuracy:.1f}%")

    # Overall accuracy
    overall_accuracy = correct_chars / total_chars * 100
    print(f"\n{'=' * 60}")
    print(f"Overall Character Accuracy: {overall_accuracy:.1f}%")
    print(f"Correct characters: {correct_chars}/{total_chars}")


def main():
    parser = argparse.ArgumentParser(description='Turkish diacritics restoration inference')
    parser.add_argument('--model', type=str, default='models/diacritics_model.pth',
                       help='Path to trained model')
    parser.add_argument('--input', type=str,
                       help='Input text file to process')
    parser.add_argument('--output', type=str,
                       help='Output file for restored text')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run benchmark on test sentences')
    parser.add_argument('--text', type=str,
                       help='Direct text input to restore')

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found.")
        print("Please train a model first using train_model.py")
        sys.exit(1)

    if args.interactive:
        interactive_mode(args.model)
    elif args.benchmark:
        benchmark_model(args.model)
    elif args.input and args.output:
        restore_text_file(args.input, args.output, args.model)
    elif args.text:
        # Direct text restoration
        restorer = DiacriticsRestorer(model_path=args.model)
        mapper = TurkishDiacriticsMapper()

        stripped = mapper.remove_diacritics(args.text)
        restored = restorer.restore_diacritics(stripped)

        print(f"Input:    {stripped}")
        print(f"Restored: {restored}")
    else:
        print("Please specify either --interactive, --benchmark, --text, or --input/--output")
        parser.print_help()


if __name__ == "__main__":
    main()