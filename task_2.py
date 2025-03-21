import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from pymorphy2 import MorphAnalyzer
import nltk

nltk.download("punkt")
nltk.download("stopwords")

# Папка с загруженными страницами
INPUT_DIR = "downloaded_pages"
TOKENS_DIR = "tokens"
LEMMAS_DIR = "lemmas"

# Создание папок для токенов и лемм
os.makedirs(TOKENS_DIR, exist_ok=True)
os.makedirs(LEMMAS_DIR, exist_ok=True)

def is_valid_token(token):
    """Проверяет, является ли слово валидным токеном."""
    return token.isalpha() and token.lower() not in stopwords.words("russian")

def process_text(text, morph):
    """Токенизирует, фильтрует и лемматизирует текст."""
    tokens = word_tokenize(text, language="russian")
    filtered_tokens = set(filter(is_valid_token, tokens))

    lemmas = defaultdict(set)
    for token in filtered_tokens:
        lemma = morph.parse(token)[0].normal_form
        lemmas[lemma].add(token)

    return sorted(filtered_tokens), lemmas

def main():
    morph = MorphAnalyzer()

    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            tokens, lemmas = process_text(text, morph)

        # Формирование имен файлов
        name, _ = os.path.splitext(filename)
        tokens_file = os.path.join(TOKENS_DIR, f"{name}_tokens.txt")
        lemmas_file = os.path.join(LEMMAS_DIR, f"{name}_lemmas.txt")

        # Запись токенов
        with open(tokens_file, "w", encoding="utf-8") as f:
            f.write("\n".join(tokens))

        # Запись лемматизированных слов
        with open(lemmas_file, "w", encoding="utf-8") as f:
            for lemma, forms in sorted(lemmas.items()):
                f.write(f"{lemma}: {' '.join(sorted(forms))}\n")

        print(f"Файлы {tokens_file} и {lemmas_file} успешно созданы.")

if __name__ == "__main__":
    main()
