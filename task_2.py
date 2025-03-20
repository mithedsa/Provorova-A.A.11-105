import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from pymorphy2 import MorphAnalyzer
import nltk
nltk.download('punkt_tab')


nltk.download("punkt")
nltk.download("stopwords")

# Папка с загруженными страницами
INPUT_DIR = "downloaded_pages"
TOKENS_FILE = "tokens_1.txt"
LEMMAS_FILE = "lemmas_1.txt"


def is_valid_token(token):
    """Проверяет, является ли слово валидным токеном."""
    return (
            token.isalpha() and  # Слово состоит только из букв
            token.lower() not in stopwords.words("russian")  # Не является стоп-словом
    )


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
    all_tokens = set()
    all_lemmas = defaultdict(set)

    for filename in os.listdir(INPUT_DIR):
        file_path = os.path.join(INPUT_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            tokens, lemmas = process_text(text, morph)
            all_tokens.update(tokens)
            for lemma, forms in lemmas.items():
                all_lemmas[lemma].update(forms)

    # Запись токенов
    with open(TOKENS_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(all_tokens)))

    # Запись лемматизированных слов
    with open(LEMMAS_FILE, "w", encoding="utf-8") as f:
        for lemma, forms in sorted(all_lemmas.items()):
            f.write(f"{lemma}: {' '.join(sorted(forms))}\n")

    print("Файлы tokens_1.txt и lemmas_1.txt успешно созданы.")


if __name__ == "__main__":
    main()
