#!/usr/bin/env python3
"""
Example usage of the Turkish Diacritics Restoration system
"""

from diacritics_restoration import DiacriticsRestorer, TurkishDiacriticsMapper
from pathlib import Path


def example_basic_restoration():
    """Basic example of restoring diacritics"""
    print("=" * 60)
    print("BASIC DIACRITICS RESTORATION")
    print("=" * 60)

    # Check if model exists
    model_path = "models/wiki_diacritics_model.pth"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first using: python train_wiki_model.py")
        return

    # Initialize restorer
    restorer = DiacriticsRestorer(model_path=model_path)

    # Example texts without diacritics
    test_texts = [
        "Bugun hava cok guzel ve gunesliydi",
        "Turkiye'nin baskenti Ankara'dir",
        "Ogrenciler universitede calisiyorlar",
        "Istanbulun tarihi cok eskidir",
        "Turk kahvesi dunyada meshurdur"
    ]

    print("\nRestoring diacritics in sample texts:\n")

    for text in test_texts:
        restored = restorer.restore_diacritics(text)
        print(f"Input:    {text}")
        print(f"Restored: {restored}")
        print()


def example_file_processing():
    """Example of processing a text file"""
    print("=" * 60)
    print("FILE PROCESSING EXAMPLE")
    print("=" * 60)

    # Create a sample input file
    input_file = "sample_input.txt"
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write("""Turkiye Cumhuriyeti'nin kurucusu Mustafa Kemal Ataturk'tur.
Ataturk, modern Turkiye'nin mimarı olarak bilinir.
Turk Dil Kurumu, Turkcenin gelisimi icin calismaktadir.
Istanbul Bogazı, Asya ve Avrupa'yi birbirine baglar.
Kapadokya'daki peri bacalari her yil binlerce turist ceker.
Turk mutfagı, zengin cesitliligiyle dunyanin en guzel mutfaklarindan biridir.""")

    print(f"Created sample input file: {input_file}")

    # Check if model exists
    model_path = "models/wiki_diacritics_model.pth"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first")
        return

    # Process the file
    restorer = DiacriticsRestorer(model_path=model_path)

    print("\nProcessing file...\n")

    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    restored_content = []
    for line in content.split('\n'):
        if line.strip():
            restored = restorer.restore_diacritics(line)
            restored_content.append(restored)
        else:
            restored_content.append('')

    # Save restored content
    output_file = "sample_output.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(restored_content))

    print(f"Restored text saved to: {output_file}")
    print("\nFirst few lines of restored text:")
    print('\n'.join(restored_content[:3]))


def example_batch_processing():
    """Example of batch processing multiple texts efficiently"""
    print("=" * 60)
    print("BATCH PROCESSING EXAMPLE")
    print("=" * 60)

    # Check if model exists
    model_path = "models/wiki_diacritics_model.pth"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first")
        return

    restorer = DiacriticsRestorer(model_path=model_path)

    # Batch of texts to process
    batch_texts = [
        "Cocuklar bahcede oynuyorlar",
        "Ogretmen sinifta ders anlatiyor",
        "Aksam yemeginde balik yedik",
        "Yarin hava yagmurlu olacakmis",
        "Kitap okumak cok onemlidir",
        "Muzik dinlemeyi cok seviyorum",
        "Spor yapmak saglik icin onemlidir",
        "Turkce cok zengin bir dildir",
        "Istanbul'da trafik cok yogun",
        "Deniz kenarinda yurumek guzeldir"
    ]

    print("\nProcessing batch of texts:\n")

    results = []
    for text in batch_texts:
        restored = restorer.restore_diacritics(text)
        results.append((text, restored))

    # Display results
    for original, restored in results[:5]:  # Show first 5
        print(f"Original: {original}")
        print(f"Restored: {restored}")
        print()

    print(f"Processed {len(batch_texts)} texts successfully")


def example_custom_processing():
    """Example with custom text preprocessing"""
    print("=" * 60)
    print("CUSTOM PREPROCESSING EXAMPLE")
    print("=" * 60)

    mapper = TurkishDiacriticsMapper()

    # Example: Text with mixed case and punctuation
    mixed_text = "BUGUN HAVA COK GUZEL! Cocuklar Parkta Oynuyorlar... Istanbul'DA Yasiyorum."

    print("Original text with mixed formatting:")
    print(mixed_text)

    # Normalize the text
    normalized = mapper.normalize_text(mixed_text)
    print(f"\nNormalized: {normalized}")

    # Remove existing diacritics (if any)
    stripped = mapper.remove_diacritics(normalized)
    print(f"Stripped:   {stripped}")

    # Check if model exists and restore
    model_path = "models/wiki_diacritics_model.pth"
    if Path(model_path).exists():
        restorer = DiacriticsRestorer(model_path=model_path)
        restored = restorer.restore_diacritics(stripped)
        print(f"Restored:   {restored}")
    else:
        print("\nModel not found - skipping restoration")


def example_accuracy_testing():
    """Example of testing restoration accuracy"""
    print("=" * 60)
    print("ACCURACY TESTING EXAMPLE")
    print("=" * 60)

    # Check if model exists
    model_path = "models/wiki_diacritics_model.pth"
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train a model first")
        return

    restorer = DiacriticsRestorer(model_path=model_path)
    mapper = TurkishDiacriticsMapper()

    # Test cases with known correct diacritics
    test_cases = [
        "Türkiye'nin başkenti Ankara'dır.",
        "Öğrenciler sınıfta ders çalışıyor.",
        "Güzel bir gün geçirdik.",
        "Çocuklar bahçede oynuyorlar.",
        "İstanbul çok kalabalık bir şehir."
    ]

    print("\nTesting restoration accuracy:\n")

    total_chars = 0
    correct_chars = 0

    for original in test_cases:
        # Remove diacritics
        stripped = mapper.remove_diacritics(original)

        # Restore
        restored = restorer.restore_diacritics(stripped)

        # Calculate accuracy
        char_matches = sum(1 for o, r in zip(original, restored) if o == r)
        accuracy = (char_matches / len(original)) * 100

        total_chars += len(original)
        correct_chars += char_matches

        print(f"Original:  {original}")
        print(f"Stripped:  {stripped}")
        print(f"Restored:  {restored}")
        print(f"Accuracy:  {accuracy:.1f}%")
        print()

    overall_accuracy = (correct_chars / total_chars) * 100
    print(f"Overall Accuracy: {overall_accuracy:.1f}%")


def main():
    """Run all examples"""
    examples = [
        ("Basic Restoration", example_basic_restoration),
        ("File Processing", example_file_processing),
        ("Batch Processing", example_batch_processing),
        ("Custom Preprocessing", example_custom_processing),
        ("Accuracy Testing", example_accuracy_testing)
    ]

    print("\n" + "=" * 60)
    print("TURKISH DIACRITICS RESTORATION - EXAMPLES")
    print("=" * 60)
    print("\nRunning all examples...\n")

    for name, func in examples:
        try:
            func()
            print()
        except Exception as e:
            print(f"Error in {name}: {e}")
            print()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()