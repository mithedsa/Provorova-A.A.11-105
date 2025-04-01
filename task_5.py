import os
import numpy as np
import math
from collections import defaultdict


def load_tfidf_vectors(tfidf_dir, num_docs):
    """Загружает TF-IDF вектора документов."""
    tfidf_vectors = {}
    vocabulary = set()

    for i in range(1, num_docs + 1):
        file_path = os.path.join(tfidf_dir, f"page_{i}_tfidf.txt")
        tfidf_vectors[i] = {}

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue  # Пропускаем некорректные строки
                term, idf, tfidf = parts[0], parts[1], parts[2]

                try:
                    idf = float(idf)
                    tfidf = float(tfidf)
                except ValueError:
                    continue  # Пропускаем строки, в которых числа не распарсились
                tfidf_vectors[i][term] = float(tfidf)
                vocabulary.add(term)

    return tfidf_vectors, sorted(vocabulary)


def vectorize_query(query, vocabulary, idf_terms):
    """Преобразует запрос в TF-IDF вектор."""
    words = query.lower().split()
    word_counts = defaultdict(int)
    for word in words:
        word_counts[word] += 1

    total_terms = sum(word_counts.values())
    query_vector = np.zeros(len(vocabulary))

    for i, term in enumerate(vocabulary):
        if term in word_counts:
            tf = word_counts[term] / total_terms
            idf = idf_terms.get(term, 0)
            query_vector[i] = tf * idf

    return query_vector


def cosine_similarity(vec1, vec2):
    """Вычисляет косинусное сходство между двумя векторами."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)


def search(query, tfidf_vectors_tokens, tfidf_vectors_lemmas, vocabulary, idf_terms):
    """Выполняет поиск по индексу и возвращает релевантные документы."""
    query_vector = vectorize_query(query, vocabulary, idf_terms)
    results = []

    for doc_id in tfidf_vectors_tokens.keys():
        doc_vector = np.zeros(len(vocabulary))
        for i, term in enumerate(vocabulary):
            doc_vector[i] = tfidf_vectors_tokens.get(doc_id, {}).get(term, 0) + tfidf_vectors_lemmas.get(doc_id,
                                                                                                         {}).get(term,
                                                                                                                 0)

        similarity = cosine_similarity(query_vector, doc_vector)
        results.append((doc_id, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


if __name__ == "__main__":
    num_docs = 100
    tokens_dir = "output_tokens"
    lemmas_dir = "output_lemmas"

    # Загружаем TF-IDF индексы для токенов и лемм
    tfidf_vectors_tokens, vocabulary_tokens = load_tfidf_vectors(tokens_dir, num_docs)
    tfidf_vectors_lemmas, vocabulary_lemmas = load_tfidf_vectors(lemmas_dir, num_docs)

    vocabulary = sorted(set(vocabulary_tokens) | set(vocabulary_lemmas))
    idf_terms = {term: math.log(num_docs / (1 + sum(
        term in tfidf_vectors_tokens[d] or term in tfidf_vectors_lemmas[d] for d in range(1, num_docs + 1)))) for term
                 in vocabulary}

    while True:
        query = input("Введите поисковый запрос: ")
        if query.lower() == "exit":
            break

        results = search(query, tfidf_vectors_tokens, tfidf_vectors_lemmas, vocabulary, idf_terms)
        print("Результаты поиска:")
        for doc_id, score in results[:10]:
            print(f"Документ {doc_id}: {score:.4f}")