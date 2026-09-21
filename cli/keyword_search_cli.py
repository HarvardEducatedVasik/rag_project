import argparse

from lib.keyword_search import search_command,build_command,tf_command,idf_command, tfidf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("build", help="Build inverted search")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("tf", help="Get term frequency of a specific document")
    search_parser.add_argument("doc_id", type=int, help="Document id of the document you want")
    search_parser.add_argument("tf_term", type=str, help="Ammount of terms used in the doc")

    search_parser = subparsers.add_parser("idf", help="Get the inverse frequency of a term")
    search_parser.add_argument("idf_term", type=str, help="Term you want to get the idf from")

    search_parser = subparsers.add_parser("tfidf", help="Get the tf-idf of a term")

    search_parser.add_argument("tfidf_doc_id", type=str, help="Term you want to get the tf-idf from")
    search_parser.add_argument("tfidf_term", type=str, help="Term you want to get the tf-idf from")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. {res['title']}")
        case "build":
            docs = build_command()
        case "tf":
            print(tf_command(args.doc_id, args.tf_term))
        case "idf":
            idf =  idf_command(args.idf_term)
            print(f"Inverse document frequency of '{args.idf_term}': {idf:.2f}")
        case "tfidf" :
            tf_idf = tfidf_command(args.tfidf_doc_id, args.tfidf_term)
            print(f"TF-IDF score of '{args.tfidf_term}' in document '{args.tfidf_doc_id}': {tf_idf}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
