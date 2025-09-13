#!/usr/bin/env python3
"""
Inference CLI for nokta-ai
"""

import argparse
import sys
from pathlib import Path
from ..core import DiacriticsRestorer, TurkishDiacriticsMapper


def interactive_mode(model_path: str):
    """Interactive diacritics restoration"""
    restorer = DiacriticsRestorer(model_path=model_path)
    mapper = TurkishDiacriticsMapper()

    print("nokta-ai Interactive Mode")
    print("=" * 30)
    print("Enter Turkish text without diacritics (Ctrl+C to exit)")
    print("Example: 'Bugun hava cok guzel' -> 'Bugün hava çok güzel'")
    print()

    try:
        while True:
            text = input("Input: ").strip()
            if not text:
                continue

            # Remove any existing diacritics first
            stripped = mapper.remove_diacritics(text)
            if stripped != text:
                print(f"Stripped: {stripped}")

            # Restore diacritics
            restored = restorer.restore_diacritics(stripped)
            print(f"Output: {restored}")
            print()

    except KeyboardInterrupt:
        print("\nGoodbye!")


def process_file(model_path: str, input_file: str, output_file: str = None):
    """Process a text file"""
    if not Path(input_file).exists():
        print(f"Error: Input file {input_file} not found")
        sys.exit(1)

    restorer = DiacriticsRestorer(model_path=model_path)

    print(f"Processing {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Process line by line for better results
    lines = content.split('\n')
    restored_lines = []

    for i, line in enumerate(lines):
        if line.strip():
            restored = restorer.restore_diacritics(line)
            restored_lines.append(restored)
        else:
            restored_lines.append(line)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(lines)} lines")

    restored_content = '\n'.join(restored_lines)

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(restored_content)
        print(f"Restored text saved to {output_file}")
    else:
        print("\nRestored text:")
        print("=" * 50)
        print(restored_content[:1000] + ("..." if len(restored_content) > 1000 else ""))


def process_text(model_path: str, text: str):
    """Process direct text input"""
    restorer = DiacriticsRestorer(model_path=model_path)
    mapper = TurkishDiacriticsMapper()

    # Remove any existing diacritics
    stripped = mapper.remove_diacritics(text)

    # Restore diacritics
    restored = restorer.restore_diacritics(stripped)

    print(f"Input:  {stripped}")
    print(f"Output: {restored}")


def main():
    """Main inference CLI"""
    parser = argparse.ArgumentParser(
        prog='nokta-inference',
        description='Turkish diacritics restoration inference'
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--text', type=str,
                            help='Direct text input')
    input_group.add_argument('--file', type=str,
                            help='Input text file')
    input_group.add_argument('--interactive', action='store_true',
                            help='Interactive mode')

    parser.add_argument('--output', type=str,
                       help='Output file (for --file mode)')

    args = parser.parse_args()

    # Check model exists
    if not Path(args.model).exists():
        print(f"Error: Model file {args.model} not found")
        sys.exit(1)

    # Execute based on mode
    if args.interactive:
        interactive_mode(args.model)
    elif args.file:
        process_file(args.model, args.file, args.output)
    elif args.text:
        process_text(args.model, args.text)
    else:
        # Default to interactive if no input specified
        interactive_mode(args.model)


if __name__ == "__main__":
    main()