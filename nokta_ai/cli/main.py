#!/usr/bin/env python3
"""
Main CLI entry point for nokta-ai
"""

import argparse
import sys
from pathlib import Path
from ..core import DiacriticsRestorer, TurkishDiacriticsMapper


def restore_text(args):
    """Restore diacritics in text"""
    if not args.model:
        print("Error: --model path is required for text restoration")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found")
        sys.exit(1)

    # Load model (constrained model is now the default)
    if args.context_size:
        print(f"Using context size: {args.context_size}")
        restorer = DiacriticsRestorer(model_path=args.model, context_size=args.context_size)
    else:
        print("Using context size from checkpoint")
        restorer = DiacriticsRestorer(model_path=args.model)

    if args.text:
        # Direct text input
        restored = restorer.restore_diacritics(args.text)
        print(restored)
    elif args.input:
        # File input
        if not Path(args.input).exists():
            print(f"Error: Input file {args.input} not found")
            sys.exit(1)

        with open(args.input, 'r', encoding='utf-8') as f:
            text = f.read()

        restored = restorer.restore_diacritics(text)

        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(restored)
            print(f"Restored text saved to {args.output}")
        else:
            print(restored)
    else:
        # Interactive mode
        print("Interactive mode - Enter text to restore diacritics (Ctrl+C to exit)")
        print("Example: 'Bugun hava cok guzel' -> 'Bugün hava çok güzel'")
        print()

        try:
            while True:
                text = input("Input: ").strip()
                if not text:
                    continue
                restored = restorer.restore_diacritics(text)
                print(f"Output: {restored}")
                print()
        except KeyboardInterrupt:
            print("\nGoodbye!")


def benchmark_model(args):
    """Benchmark model performance"""
    if not args.model:
        print("Error: --model path is required for benchmarking")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found")
        sys.exit(1)

    restorer = DiacriticsRestorer(model_path=args.model)
    mapper = TurkishDiacriticsMapper()

    # Test sentences
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

    print("Benchmarking nokta-ai model")
    print("=" * 50)

    total_chars = 0
    correct_chars = 0

    for stripped, expected in test_sentences:
        input_text = mapper.remove_diacritics(stripped)
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

    overall_accuracy = correct_chars / total_chars * 100
    print(f"\n{'=' * 50}")
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")
    print(f"Correct characters: {correct_chars}/{total_chars}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='nokta',
        description='Turkish Diacritics Restoration with Neural Networks'
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore diacritics in text')
    restore_parser.add_argument('--model', type=str, required=True,
                               help='Path to trained model file')
    restore_parser.add_argument('--text', type=str,
                               help='Text to restore (direct input)')
    restore_parser.add_argument('--input', type=str,
                               help='Input file path')
    restore_parser.add_argument('--output', type=str,
                               help='Output file path')
    restore_parser.add_argument('--context-size', type=int,
                               help='Override context size (default: use value from model checkpoint)')
    restore_parser.set_defaults(func=restore_text)

    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Benchmark model performance')
    benchmark_parser.add_argument('--model', type=str, required=True,
                                 help='Path to trained model file')
    benchmark_parser.set_defaults(func=benchmark_model)

    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate on test dataset')
    evaluate_parser.add_argument('--model', type=str, required=True,
                                help='Path to trained model file')
    evaluate_parser.add_argument('--test-file', type=str, required=True,
                                help='Path to test file (e.g., data/test_datasets/vikipedi_test.txt)')
    evaluate_parser.add_argument('--output', type=str,
                                help='Save detailed results to file')
    evaluate_parser.add_argument('--context-size', type=int,
                                help='Override context size (default: use value from model checkpoint)')
    evaluate_parser.add_argument('--num-passes', type=int, default=1,
                                help='Number of restoration passes (default: 1). Multiple passes can improve restoration of words with multiple diacritics.')

    def evaluate_command(args):
        from .evaluate_constrained import evaluate_constrained_model
        evaluate_constrained_model(args.model, args.test_file, args.output, args.context_size, args.num_passes)

    evaluate_parser.set_defaults(func=evaluate_command)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


def restore_main():
    """Entry point for nokta-restore command"""
    parser = argparse.ArgumentParser(
        description='Restore Turkish diacritics in text using trained neural network model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  nokta-restore --model models/my_model.pth

  # Direct text input
  nokta-restore --model models/my_model.pth --text "Bugun hava cok guzel"

  # Process a file
  nokta-restore --model models/my_model.pth --input input.txt --output output.txt
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--text', type=str,
                       help='Text to restore (direct input)')
    parser.add_argument('--input', type=str,
                       help='Input file path')
    parser.add_argument('--output', type=str,
                       help='Output file path')
    parser.add_argument('--context-size', type=int,
                       help='Override context size (default: use value from model checkpoint)')

    args = parser.parse_args()
    restore_text(args)


if __name__ == "__main__":
    main()