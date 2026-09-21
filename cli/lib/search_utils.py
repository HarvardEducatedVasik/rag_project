import json
import os
import pickle
import string
from typing import Any, TypedDict
from collections import Counter
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()



DEFAULT_ALPHA = 0.5
RRF_K = 60
SEARCH_MULTIPLIER = 5

DEFAULT_SEARCH_LIMIT = 5
DOCUMENT_PREVIEW_LENGTH = 100
SCORE_PRECISION = 3

BM25_K1 = 1.5
BM25_B = 0.75

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
GOLDEN_DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "golden_dataset.json")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 1
DEFAULT_SEMANTIC_CHUNK_SIZE = 4

MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")
CHUNK_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
CHUNK_METADATA_PATH = os.path.join(CACHE_DIR, "chunk_metadata.json")

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def load_stopwords() -> list[str]:
    with open("./data/stopwords.txt", "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]
    
STOPWORDS = load_stopwords()


def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

movies = load_movies()

def tokenize_term(term : str):
    token = tokenize_text(term)
    if len(token) != 1:
        raise Exception("Term cannot be tokenized into one token")
    return token[0]


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    filtered_words = []
    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(stemmer.stem(word))
    return filtered_words

class Movie(TypedDict):
    id: int
    title: str
    description: str

class InvertedIndex():
    def __init__(self):
        self.index : dict[str, set[int]] = {}
        self.docmap : dict[int, Movie] = {}
        self.term_frequencies : dict[int, Counter] = {}
    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text) 
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            if doc_id not in self.term_frequencies:
                self.term_frequencies[doc_id] = Counter()
            self.term_frequencies[doc_id][token]+= 1
            self.index[token].add(doc_id)


    def get_documents(self, term):
        final_term = tokenize_text(term)[0]
        ids = self.index.get(final_term, set())
        return sorted(list(ids))
    def build(self):
        movies = load_movies()
        for movie in movies:
            self.docmap[movie["id"]]=movie
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
    def save (self):
        if not os.path.exists("cache"):
            os.makedirs("cache")
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)
        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)
    def load(self):
        try :
            with open ("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)    
        except Exception as e:
            raise Exception("files not found + " , e)
    def get_tf(self, doc_id, term):
        try :
            token = tokenize_term(term)
            return self.term_frequencies[token]
        except Exception as e:
            print(e)
        



class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]


class GoldenTestCase(TypedDict):
    query: str
    relevant_docs: list[str]


class GoldenDataset(TypedDict):
    test_cases: list[GoldenTestCase]






def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }


def load_golden_dataset() -> GoldenDataset:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)
