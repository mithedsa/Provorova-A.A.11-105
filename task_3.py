import os
import re
from collections import defaultdict

INDEX_FILE = "inverted_index.txt"

# Папка с обработанными леммами
LEMMAS_DIR = "lemmas"


def build_inverted_index():
    inverted_index = defaultdict(set)

    for filename in os.listdir(LEMMAS_DIR):
        file_path = os.path.join(LEMMAS_DIR, filename)
        if not os.path.isfile(file_path):
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                lemma, *forms = line.strip().split(": ")
                if forms:
                    inverted_index[lemma].add(filename)

    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        for lemma, files in sorted(inverted_index.items()):
            f.write(f"{lemma}: {','.join(files)}\n")

    return inverted_index


def boolean_search(query, index):
    def parse_term(term):
        return index.get(term, set())

    def eval_not(term):
        all_docs = set(os.listdir(LEMMAS_DIR))
        return all_docs - parse_term(term)

    def eval_expression(expression):
        expression = re.sub(r'\bAND\b', '&', expression)
        expression = re.sub(r'\bOR\b', '|', expression)
        expression = re.sub(r'\bNOT\b', '~', expression)

        tokens = re.split(r'([()&|~])', expression)
        tokens = [t.strip() for t in tokens if t.strip()]

        stack = []
        for token in tokens:
            if token in ('&', '|', '(', ')'):
                stack.append(token)
            elif token.startswith('~'):
                stack.append(eval_not(token[1:]))
            else:
                stack.append(parse_term(token))

        return eval(" ".join(map(str, stack)))

    return eval_expression(query)


if __name__ == "__main__":
    index = build_inverted_index()
    while True:
        user_query = input("Введите поисковый запрос: ").strip()
        if user_query.lower() == "exit":
            break
        result = boolean_search(user_query, index)
        print("Результаты поиска:", result)
