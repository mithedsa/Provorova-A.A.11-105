import os
import math
from collections import Counter


def read_tokens(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]


def read_lemmas(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip().split(": ")[1] for line in f]


def compute_tf(term_counts, total_terms):
    return {term: count / total_terms for term, count in term_counts.items()}


def compute_idf(doc_term_counts, num_docs):
    idf = {}
    for term, doc_count in doc_term_counts.items():
        idf[term] = math.log(num_docs / (1 + doc_count))
    return idf


def process_files():
    downloaded_pages = "downloaded_pages"
    tokens_dir = "tokens"
    lemmas_dir = "lemmas"
    output_tokens_dir = "output_tokens"
    output_lemmas_dir = "output_lemmas"
    os.makedirs(output_tokens_dir, exist_ok=True)
    os.makedirs(output_lemmas_dir, exist_ok=True)

    num_docs = 100
    doc_term_counts = Counter()
    doc_lemma_counts = Counter()
    term_frequencies = {}
    lemma_frequencies = {}

    for i in range(1, num_docs + 1):
        token_file = os.path.join(tokens_dir, f"page_{i}_tokens.txt")
        lemma_file = os.path.join(lemmas_dir, f"page_{i}_lemmas.txt")

        tokens = read_tokens(token_file)
        lemmas = read_lemmas(lemma_file)

        token_counts = Counter(tokens)
        lemma_counts = Counter(lemmas)

        doc_term_counts.update(set(tokens))
        doc_lemma_counts.update(set(lemmas))

        total_terms = sum(token_counts.values())
        total_lemmas = sum(lemma_counts.values())

        term_frequencies[i] = compute_tf(token_counts, total_terms)
        lemma_frequencies[i] = compute_tf(lemma_counts, total_lemmas)

    idf_terms = compute_idf(doc_term_counts, num_docs)
    idf_lemmas = compute_idf(doc_lemma_counts, num_docs)

    for i in range(1, num_docs + 1):
        token_output_file = os.path.join(output_tokens_dir, f"page_{i}_tfidf.txt")
        lemma_output_file = os.path.join(output_lemmas_dir, f"page_{i}_tfidf.txt")

        with open(token_output_file, "w", encoding="utf-8") as f:
            for term, tf in term_frequencies[i].items():
                tfidf = tf * idf_terms[term]
                f.write(f"{term} {idf_terms[term]:.6f} {tfidf:.6f}\n")

        with open(lemma_output_file, "w", encoding="utf-8") as f:
            for lemma, tf in lemma_frequencies[i].items():
                tfidf = tf * idf_lemmas[lemma]
                f.write(f"{lemma} {idf_lemmas[lemma]:.6f} {tfidf:.6f}\n")


if __name__ == "__main__":
    process_files()
