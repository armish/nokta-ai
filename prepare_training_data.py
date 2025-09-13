#!/usr/bin/env python3
"""
Prepare training data for Turkish diacritics restoration
Supports various data sources including Wikipedia dumps and text files
"""

import os
import json
import pickle
import random
from pathlib import Path
from typing import List, Tuple, Generator
import requests
from tqdm import tqdm


def download_turkish_wikipedia_sample(output_dir: str = "data", num_articles: int = 1000):
    """Download sample Turkish Wikipedia articles using the API"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "wikipedia_sample.txt"

    print(f"Downloading {num_articles} Turkish Wikipedia articles...")

    articles = []
    base_url = "https://tr.wikipedia.org/w/api.php"

    # Get random articles
    params = {
        'action': 'query',
        'format': 'json',
        'list': 'random',
        'rnlimit': min(20, num_articles),  # API limit
        'rnnamespace': 0  # Main namespace only
    }

    collected = 0
    with tqdm(total=num_articles) as pbar:
        while collected < num_articles:
            try:
                response = requests.get(base_url, params=params)
                data = response.json()

                for page in data['query']['random']:
                    page_id = page['id']

                    # Get page content
                    content_params = {
                        'action': 'query',
                        'format': 'json',
                        'pageids': page_id,
                        'prop': 'extracts',
                        'exintro': True,
                        'explaintext': True
                    }

                    content_response = requests.get(base_url, params=content_params)
                    content_data = content_response.json()

                    page_content = content_data['query']['pages'][str(page_id)]
                    if 'extract' in page_content:
                        text = page_content['extract']
                        if len(text) > 100:  # Filter short articles
                            articles.append(text)
                            collected += 1
                            pbar.update(1)

                            if collected >= num_articles:
                                break

            except Exception as e:
                print(f"Error fetching articles: {e}")
                continue

    # Save articles
    with open(output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(article + '\n\n')

    print(f"Saved {len(articles)} articles to {output_file}")
    return str(output_file)


def load_text_files(file_paths: List[str]) -> Generator[str, None, None]:
    """Load text from files"""
    for file_path in file_paths:
        path = Path(file_path)
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                yield f.read()
        else:
            print(f"Warning: File {file_path} not found")


def prepare_training_validation_split(texts: List[str],
                                     train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """Split texts into training and validation sets"""
    random.shuffle(texts)
    split_idx = int(len(texts) * train_ratio)
    return texts[:split_idx], texts[split_idx:]


def create_synthetic_corpus(output_dir: str = "data", num_sentences: int = 10000):
    """Create a synthetic corpus with common Turkish words and patterns"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = Path(output_dir) / "synthetic_corpus.txt"

    # Common Turkish words with diacritics
    words_with_diacritics = [
        # Common nouns
        "çocuk", "öğretmen", "öğrenci", "üniversite", "şehir", "ülke", "dünya",
        "gün", "yıl", "süre", "kültür", "müze", "köprü", "göl", "dağ",
        "güneş", "yağmur", "rüzgar", "çiçek", "ağaç", "kuş", "köpek",

        # Common verbs
        "gitmek", "gelmek", "görmek", "düşünmek", "çalışmak", "öğrenmek",
        "üretmek", "değiştirmek", "büyümek", "küçülmek", "gülmek", "ağlamak",

        # Common adjectives
        "büyük", "küçük", "güzel", "çirkin", "yüksek", "düşük", "önemli",
        "gerçek", "özel", "genel", "açık", "kapalı", "soğuk", "sıcak",

        # Common proper nouns
        "Türkiye", "İstanbul", "Ankara", "İzmir", "Antalya", "Bursa",
        "Atatürk", "Cumhuriyet", "Avrupa", "Asya", "Akdeniz", "Karadeniz",

        # Common expressions
        "teşekkür", "günaydın", "görüşürüz", "hoşgeldiniz", "güle güle"
    ]

    # Sentence patterns
    patterns = [
        "{noun} çok {adjective}.",
        "{name} {verb} istiyor.",
        "Bu {noun} {adjective} görünüyor.",
        "{name}'de {noun} var.",
        "{noun} ve {noun} {adjective}.",
        "Dün {name}'ye gittim.",
        "{noun} için {verb} gerekiyor.",
        "{adjective} bir {noun} gördüm.",
        "{name}'nin {noun}ı {adjective}.",
        "Her gün {verb} önemlidir."
    ]

    sentences = []
    for _ in range(num_sentences):
        pattern = random.choice(patterns)
        sentence = pattern

        # Replace placeholders
        replacements = {
            "{noun}": random.choice([w for w in words_with_diacritics if w[0].islower()]),
            "{verb}": random.choice(["gitmek", "gelmek", "görmek", "düşünmek", "çalışmak",
                                    "öğrenmek", "üretmek", "değiştirmek"]),
            "{adjective}": random.choice(["büyük", "küçük", "güzel", "önemli", "özel",
                                         "yüksek", "düşük", "soğuk", "sıcak"]),
            "{name}": random.choice(["Türkiye", "İstanbul", "Ankara", "İzmir", "Antalya"])
        }

        for placeholder, value in replacements.items():
            if placeholder in sentence:
                sentence = sentence.replace(placeholder, value, 1)

        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:]
        sentences.append(sentence)

    # Group into paragraphs
    paragraphs = []
    for i in range(0, len(sentences), 5):
        paragraph = ' '.join(sentences[i:i+5])
        paragraphs.append(paragraph)

    # Save corpus
    with open(output_file, 'w', encoding='utf-8') as f:
        for paragraph in paragraphs:
            f.write(paragraph + '\n\n')

    print(f"Created synthetic corpus with {num_sentences} sentences in {output_file}")
    return str(output_file)


def prepare_dataset_cache(text_files: List[str], cache_file: str = "data/dataset_cache.pkl"):
    """Prepare and cache the dataset for faster loading"""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)

    all_texts = []
    for text in load_text_files(text_files):
        # Split into sentences or paragraphs
        paragraphs = text.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 50:  # Filter short paragraphs
                all_texts.append(para.strip())

    # Save cache
    with open(cache_file, 'wb') as f:
        pickle.dump(all_texts, f)

    print(f"Cached {len(all_texts)} text samples to {cache_file}")
    return cache_file


def load_dataset_cache(cache_file: str) -> List[str]:
    """Load cached dataset"""
    with open(cache_file, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)

    print("Preparing Turkish text data for diacritics restoration training")
    print("=" * 60)

    # 1. Create synthetic corpus for quick testing
    synthetic_file = create_synthetic_corpus(num_sentences=5000)

    # 2. Optional: Download Wikipedia sample (uncomment to use)
    # wiki_file = download_turkish_wikipedia_sample(num_articles=100)

    # 3. Prepare dataset cache
    text_files = [synthetic_file]
    # Add more files if available:
    # text_files.append(wiki_file)
    # text_files.append("path/to/your/turkish_text.txt")

    cache_file = prepare_dataset_cache(text_files)

    # 4. Load and split data
    texts = load_dataset_cache(cache_file)
    train_texts, val_texts = prepare_training_validation_split(texts)

    print(f"\nDataset prepared:")
    print(f"  Training samples: {len(train_texts)}")
    print(f"  Validation samples: {len(val_texts)}")
    print(f"  Cache file: {cache_file}")

    # Show sample
    print("\nSample text:")
    print(train_texts[0][:200] + "...")