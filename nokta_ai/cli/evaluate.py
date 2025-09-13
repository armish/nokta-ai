#!/usr/bin/env python3
"""
Evaluation CLI for nokta-ai using test datasets
"""

import argparse
import sys
from pathlib import Path
from ..core import DiacriticsRestorer, TurkishDiacriticsMapper


def evaluate_on_file(model_path: str, test_file: str, output_file: str = None):
    """Evaluate model on a test file with ground truth"""
    if not Path(test_file).exists():
        print(f"Error: Test file {test_file} not found")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"Error: Model file {model_path} not found")
        sys.exit(1)

    # Load model
    print(f"Loading model from {model_path}...")
    restorer = DiacriticsRestorer(model_path=model_path)
    mapper = TurkishDiacriticsMapper()

    # Load test sentences
    with open(test_file, 'r', encoding='utf-8') as f:
        test_sentences = [line.strip() for line in f if line.strip()]

    print(f"Evaluating on {len(test_sentences)} sentences from {test_file}")
    print("=" * 60)

    total_chars = 0
    correct_chars = 0
    total_words = 0
    correct_words = 0
    results = []

    for i, original in enumerate(test_sentences):
        # Remove diacritics to create input
        input_text = mapper.remove_diacritics(original)

        # Skip if no diacritics to restore
        if input_text == original:
            continue

        # Restore diacritics
        restored = restorer.restore_diacritics(input_text)

        # Character-level accuracy
        char_matches = sum(1 for c1, c2 in zip(original, restored) if c1 == c2)
        char_accuracy = (char_matches / len(original)) * 100

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

        # Store result
        result = {
            'original': original,
            'input': input_text,
            'restored': restored,
            'char_accuracy': char_accuracy,
            'word_accuracy': word_accuracy
        }
        results.append(result)

        # Print progress for first few and every 10th
        if i < 5 or (i + 1) % 10 == 0:
            print(f"\nSentence {i+1}:")
            print(f"Original:  {original}")
            print(f"Input:     {input_text}")
            print(f"Restored:  {restored}")
            print(f"Char Acc:  {char_accuracy:.1f}%")
            print(f"Word Acc:  {word_accuracy:.1f}%")

    # Calculate overall accuracy
    overall_char_accuracy = (correct_chars / total_chars) * 100 if total_chars > 0 else 0
    overall_word_accuracy = (correct_words / total_words) * 100 if total_words > 0 else 0

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Test file: {test_file}")
    print(f"Sentences processed: {len(results)}")
    print(f"Total characters: {total_chars}")
    print(f"Total words: {total_words}")
    print(f"Overall character accuracy: {overall_char_accuracy:.2f}%")
    print(f"Overall word accuracy: {overall_word_accuracy:.2f}%")

    # Save detailed results if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Evaluation Results - {test_file}\n")
            f.write("=" * 60 + "\n\n")

            for i, result in enumerate(results):
                f.write(f"Sentence {i+1}:\n")
                f.write(f"Original:  {result['original']}\n")
                f.write(f"Input:     {result['input']}\n")
                f.write(f"Restored:  {result['restored']}\n")
                f.write(f"Char Acc:  {result['char_accuracy']:.1f}%\n")
                f.write(f"Word Acc:  {result['word_accuracy']:.1f}%\n\n")

            f.write("=" * 60 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 60 + "\n")
            f.write(f"Overall character accuracy: {overall_char_accuracy:.2f}%\n")
            f.write(f"Overall word accuracy: {overall_word_accuracy:.2f}%\n")

        print(f"Detailed results saved to: {output_file}")

    return overall_char_accuracy, overall_word_accuracy


def main():
    """Main evaluation CLI"""
    parser = argparse.ArgumentParser(
        prog='nokta-evaluate',
        description='Evaluate nokta-ai model on test datasets'
    )

    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--test-file', type=str, required=True,
                       help='Path to test file with ground truth')
    parser.add_argument('--output', type=str,
                       help='Save detailed results to file')

    args = parser.parse_args()

    # Run evaluation
    char_acc, word_acc = evaluate_on_file(args.model, args.test_file, args.output)

    # Exit with status code based on accuracy
    if char_acc >= 90:
        sys.exit(0)  # Excellent
    elif char_acc >= 80:
        sys.exit(1)  # Good
    elif char_acc >= 70:
        sys.exit(2)  # Fair
    else:
        sys.exit(3)  # Poor


if __name__ == "__main__":
    main()