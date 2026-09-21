import string
import math

from .search_utils import *

inverted_index = InvertedIndex()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    try :
        results = []
        query_tokens = tokenize_text(query)
        inverted_index.load()
        for token in query_tokens:
            for movie_id in inverted_index.get_documents(token):
                results.append(inverted_index.docmap[movie_id])
                if len(results) >= limit:
                    return results

        return results
    except Exception as e:
        print(e)
def build_command():
    inverted_index.build()
    inverted_index.save()


def tf_command(doc_id : int, term : str):
    
    return inverted_index.term_frequencies[doc_id][term]


def idf_command (term : str):
    try:
        inverted_index.load()
        token = tokenize_term(term)
        total_doc_count = len(inverted_index.docmap)
        term_match_doc_count = 0
        for doc_id in inverted_index.docmap.keys():
            try :
                if (tf_command(doc_id, token))>=1:
                    term_match_doc_count += 1            
            except Exception as e:
                print(e)
        return (math.log((total_doc_count + 1) / (term_match_doc_count + 1)))
    except Exception as e:
        print(e)

def tfidf_command( doc_id : int ,term : str ):
    try:
        inverted_index.load()
        token = tokenize_term(term)
        print(doc_id)
        idf = idf_command(token)
        print(f"idf : {idf}")
        tf = inverted_index.term_frequencies[doc_id][token]
        print("zmrdel")
        print(f"tf : {tf}")
        if idf == None:
            idf = 0
        return tf * float(idf)
    except Exception as e:
        print(e)
    







