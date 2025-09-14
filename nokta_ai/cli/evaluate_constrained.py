#!/usr/bin/env python3
"""
Evaluation CLI for constrained nokta-ai model
"""

import argparse
import sys
from pathlib import Path
from ..models.constrained import ConstrainedDiacriticsRestorer, remove_diacritics_simple


def evaluate_constrained_model(model_path: str, test_file: str, output_file: str = None, context_size: int = None):
    """Evaluate constrained model on test file"""
    if not Path(test_file).exists():
        print(f"Error: Test file {test_file} not found")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)

    # Load model
    print(f"Loading constrained model from {model_path}...")
    if context_size:
        print(f"Using specified context size: {context_size}")
        restorer = ConstrainedDiacriticsRestorer(model_path=model_path, context_size=context_size)
    else:
        print("Using context size from model checkpoint")
        restorer = ConstrainedDiacriticsRestorer(model_path=model_path)

    # Load test sentences
    with open(test_file, 'r', encoding='utf-8') as f:
        test_sentences = [line.strip() for line in f if line.strip()]

    print(f"Evaluating constrained model on {len(test_sentences)} sentences")
    print("=" * 60)

    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0
    total_diacritic_chars = 0
    correct_diacritic_chars = 0

    sentences_with_diacritics = 0

    for i, original in enumerate(test_sentences):
        # Remove diacritics to create input
        input_text = remove_diacritics_simple(original)

        # Skip if no diacritics to restore
        if input_text == original:
            continue

        sentences_with_diacritics += 1

        # Show progress every 20 sentences after the first 5
        if i >= 5 and sentences_with_diacritics % 20 == 0:
            print(f"Processing sentence {sentences_with_diacritics}...")

        # Restore diacritics
        restored = restorer.restore_diacritics(input_text)

        # Ensure same length (should be guaranteed by constrained model)
        if len(restored) != len(original):
            print(f"WARNING: Length mismatch in sentence {i+1}")
            print(f"Original length: {len(original)}, Restored length: {len(restored)}")
            continue

        # Character-level accuracy
        char_matches = sum(1 for c1, c2 in zip(original, restored) if c1 == c2)
        char_accuracy = (char_matches / len(original)) * 100

        # Diacritic-specific accuracy (only count chars that should have diacritics)
        diacritic_matches = 0
        diacritic_total = 0

        for orig_char, rest_char, inp_char in zip(original, restored, input_text):
            if orig_char != inp_char:  # This character had a diacritic
                diacritic_total += 1
                if orig_char == rest_char:
                    diacritic_matches += 1

        diacritic_accuracy = (diacritic_matches / diacritic_total * 100) if diacritic_total > 0 else 100

        # Word-level accuracy
        orig_words = original.split()
        rest_words = restored.split()
        word_matches = sum(1 for w1, w2 in zip(orig_words, rest_words) if w1 == w2)
        word_accuracy = (word_matches / len(orig_words)) * 100 if orig_words else 0

        # Accumulate totals
        total_chars += len(original)
        correct_chars += char_matches
        total_words += len(orig_words)
        correct_words += word_matches
        total_diacritic_chars += diacritic_total
        correct_diacritic_chars += diacritic_matches

        # Print first few examples
        if i < 5:
            print(f"\nSentence {i+1}:")
            print(f"Original:   {original}")
            print(f"Input:      {input_text}")
            print(f"Restored:   {restored}")
            print(f"Char Acc:   {char_accuracy:.1f}%")
            print(f"Diacritic:  {diacritic_accuracy:.1f}% ({diacritic_matches}/{diacritic_total})")
            print(f"Word Acc:   {word_accuracy:.1f}%")

    # Calculate overall accuracy
    overall_char_accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    overall_word_accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0
    overall_diacritic_accuracy = (correct_diacritic_chars / total_diacritic_chars) * 100 if total_diacritic_chars > 0 else 0

    # Print summary
    print("\n" + "=" * 60)
    print("CONSTRAINED MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test file: {test_file}")
    print(f"Sentences processed: {sentences_with_diacritics}")
    print(f"Total characters: {total_chars}")
    print(f"Total words: {total_words}")
    print(f"Total diacritic positions: {total_diacritic_chars}")
    print()
    print(f"Overall character accuracy: {overall_char_accuracy:.2f}%")
    print(f"Overall word accuracy: {overall_word_accuracy:.2f}%")
    print(f"Diacritic-specific accuracy: {overall_diacritic_accuracy:.2f}%")
    print(f"  (This measures how well the model restores diacritics)")

    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Constrained Model Evaluation Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Overall character accuracy: {overall_char_accuracy:.2f}%\n")
            f.write(f"Overall word accuracy: {overall_word_accuracy:.2f}%\n")
            f.write(f"Diacritic-specific accuracy: {overall_diacritic_accuracy:.2f}%\n\n")

        print(f"Results saved to: {output_file}")

    return overall_char_accuracy, overall_word_accuracy, overall_diacritic_accuracy


def main():
    parser = argparse.ArgumentParser(
        prog='nokta-evaluate-constrained',
        description='Evaluate constrained nokta-ai model'
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained constrained model file')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to test file with ground truth')
    parser.add_argument('--output', type=str,
                       help='Save detailed results to file')
    parser.add_argument('--context-size', type=int,
                       help='Override context size (use model default if not specified)')

    args = parser.parse_args()

    char_acc, word_acc, diacritic_acc = evaluate_constrained_model(
        args.model, args.test_file, args.output, args.context_size
    )

    # Exit with status based on diacritic accuracy (most important metric)
    if diacritic_acc >= 90:
        sys.exit(0)  # Excellent
    elif diacritic_acc >= 80:
        sys.exit(1)  # Good
    else:
        sys.exit(2)  # Needs improvement


if __name__ == "__main__":
    main()