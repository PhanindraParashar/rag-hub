import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from langchain_core.documents import Document 

class InvertedIndex:
    """
    Creates, stores, and queries an inverted index for keyword-based search.

    The index maps keywords to a list of document locations (source file and
    page number) where that keyword appears. This provides a highly efficient
    way to retrieve all relevant pages for a given search term.
    """
    def __init__(self):
        """Initialises a new, empty inverted index."""
        # The core data structure: {keyword: [(file_name, page_num), ...]}
        # Using defaultdict for cleaner building logic.
        self.index: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # To keep track of all documents loaded into the index for easy retrieval.
        self.all_docs: Dict[str, List[Document]] = {}

    def build_from_docs(self, docs_map: Dict[str, List[Document]]):
        """
        Builds the index from scratch from a dictionary of documents.

        This method clears any existing index data before building the new one.

        Parameters
        ----------
        docs_map : Dict[str, List[Document]]
            A dictionary where keys are source filenames and values are lists
            of Document objects. Each Document's metadata must contain
            'keywords' and 'page_number'.
        """
        print("Building inverted index from scratch...")
        self.index.clear()
        self.all_docs.clear()
        self.update(docs_map) # Use the update logic to perform the build
            
        print(f"Index built successfully. Found {len(self.index)} unique keywords across {len(self.all_docs)} files.")

    def update(self, new_docs_map: Dict[str, List[Document]]):
        """
        Updates the index with new documents without rebuilding.

        If a document filename already exists in the index, its old entries
        are removed before the new ones are added.

        Parameters
        ----------
        new_docs_map : Dict[str, List[Document]]
            A dictionary of new documents to add to the index.
        """
        print(f"Updating index with {len(new_docs_map)} new document(s)...")
        for file_name, doc_list in new_docs_map.items():
            # If the document already exists, remove its old entries first
            if file_name in self.all_docs:
                self._remove_document(file_name)

            # Add new document to the main collection
            self.all_docs[file_name] = doc_list
            
            # Add new entries to the index
            for doc in doc_list:
                if "keywords" in doc.metadata and "page_number" in doc.metadata:
                    page_num = doc.metadata["page_number"]
                    for keyword in doc.metadata["keywords"]:
                        self.index[keyword].append((file_name, page_num))
                        # Keep the posting list sorted
                        self.index[keyword].sort()
        print("Update complete.")

    def _remove_document(self, file_name: str):
        """
        A private helper to remove all index entries for a given document.
        """
        print(f"Removing existing entries for '{file_name}' before update...")
        keywords_to_prune = []
        for keyword, postings in self.index.items():
            # Create a new list excluding the entries for the given file_name
            self.index[keyword] = [post for post in postings if post[0] != file_name]
            # If the list becomes empty, mark the keyword for deletion
            if not self.index[keyword]:
                keywords_to_prune.append(keyword)

        # Remove keywords that no longer have any associated documents
        for keyword in keywords_to_prune:
            del self.index[keyword]
            
        # Remove from the docs collection
        if file_name in self.all_docs:
            del self.all_docs[file_name]


    def search(self, keyword: str) -> List[Tuple[str, int]]:
        """
        Searches the index for a single keyword. The search is case-insensitive.

        Parameters
        ----------
        keyword : str
            The keyword to search for.

        Returns
        -------
        List[Tuple[str, int]]
            A list of (file_name, page_number) tuples where the keyword was found.
            Returns an empty list if the keyword is not in the index.
        """
        # Assuming keywords were indexed in lowercase during your annotation step.
        keyword = keyword.lower() 
        return self.index.get(keyword, [])

    def get_document_page(self, file_name: str, page_number: int) -> Optional[Document]:
        """
        Retrieves a specific Document object using its file name and page number.

        Parameters
        ----------
        file_name : str
            The source file name of the document.
        page_number : int
            The page number within the document (1-indexed).

        Returns
        -------
        Optional[Document]
            The Document object if found, otherwise None.
        """
        if file_name in self.all_docs:
            # This is faster than a loop if page numbers are sequential and 1-indexed
            if 1 <= page_number <= len(self.all_docs[file_name]):
                doc = self.all_docs[file_name][page_number - 1]
                if doc.metadata.get("page_number") == page_number:
                    return doc
            # Fallback for non-sequential pages
            for doc in self.all_docs[file_name]:
                 if doc.metadata.get("page_number") == page_number:
                    return doc
        return None

    def save(self, filepath: Union[str, Path]):
        """
        Saves the inverted index to a file in JSON format.

        Parameters
        ----------
        filepath : str or Path
            The path to the file where the index will be saved.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving index to {path}...")
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.index, f, indent=2)
        print("Save complete.")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'InvertedIndex':
        """
        Loads an inverted index from a JSON file.

        This is a class method, so you can call it like:
        ``my_index = InvertedIndex.load("path/to/index.json")``

        Note: This only loads the index itself, not the `all_docs` mapping.
        You should rebuild or re-associate the `all_docs` mapping after loading
        if you need to use `get_document_page`.

        Parameters
        ----------
        filepath : str or Path
            The path to the file containing the saved index.

        Returns
        -------
        InvertedIndex
            A new InvertedIndex instance populated with the loaded data.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Index file not found at {path}")
        
        print(f"Loading index from {path}...")
        instance = cls()
        with path.open("r", encoding="utf-8") as f:
            regular_dict = json.load(f)
            instance.index = defaultdict(list, regular_dict)
            
        print(f"Index loaded. Contains {len(instance.index)} unique keywords.")
        return instance
