import string


from .search_utils import *




def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    results = []
    stopwords = load_stopwords()
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results

def build_command(term : str):
    inverted_index = InvertedIndex()
    inverted_index.build()
    docs = inverted_index.get_documents(term)
    inverted_index.save()
    return docs









