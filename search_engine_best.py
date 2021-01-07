import pandas as pd
from reader import ReadFile
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
from ranker import Ranker
from localMethod import LocalMethod
from thesaurus import Thesaurus

import numpy as np


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None):
        self._config = config
        # self._parser = Parse()
        self._parser = Parse(self._config)
        self._indexer = Indexer(self._config)
        self._ranker = Ranker()
        self._model = None
        self._searcher = Searcher(self._parser, self._indexer)
        self._thesaurus = Thesaurus()

    # TODO - check if need to keep this func, all corpus
    def run_engine(self):
        """
        :return:
        """
        r = ReadFile(corpus_path=self._config.get__corpusPath())
        number_of_files = 0

        for i, file in enumerate(r.read_corpus()):
            # Iterate over every document in the file
            number_of_files += 1
            for idx, document in enumerate(file):
                # parse the document
                parsed_document = self._parser.parse_doc(document)
                self._indexer.add_new_doc(parsed_document)

        self._indexer.entities_and_small_big()
        self._indexer.calculate_idf(self._parser.number_of_documents)
        # avg_doc_len = self._parser.total_len_docs / self._parser.number_of_documents
        # self._indexer.save_index("inverted_idx")
        # TODO - check the name of inverted_idx
        self._indexer.save_index("idx_bench.pkl")

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def build_index_from_parquet(self, fn):
        """
        Reads parquet file and passes it to the parser, then indexer.
        Input:
            fn - path to parquet file
        Output:
            No output, just modifies the internal _indexer object.
        """
        df = pd.read_parquet(fn, engine="pyarrow")
        documents_list = df.values.tolist()
        # Iterate over every document in the file
        number_of_documents = 0
        for idx, document in enumerate(documents_list):
            # parse the document
            parsed_document = self._parser.parse_doc(document)
            number_of_documents += 1
            # index the document data
            self._indexer.add_new_doc(parsed_document)
        # self._indexer.entities_and_small_big()
        ###########
        self.test_and_clean()
        ###########
        self._indexer.calculate_idf(self._parser.number_of_documents)
        self._indexer.save_index("idx_bench.pkl")
        print('Finished parsing and indexing.')

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        return self._indexer.load_index(fn)

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_precomputed_model(self, model_dir=None):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and
        assign to self._model, which is passed on to the searcher at query time.
        """
        pass

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """

        query_as_list = self._parser.parse_sentence(query)[0]
        query_dict, max_tf_query = self._searcher.get_query_dict(query_as_list)

        thes = Thesaurus()
        expanded_query = thes.expand_query(query_dict, max_tf_query)
        round_1_len, round_1 = self.search_helper(expanded_query, 100, 0)

        local = LocalMethod(self._indexer)
        expanded_query_dict = local.expand_query(query_dict, max_tf_query, round_1)

        return self.search_helper(expanded_query_dict, None, 0.3)

    def search_helper(self, query_dict, k, p=0):
        relevant_docs, query_vector = self._searcher.relevant_docs_from_posting(query_dict, p)
        n_relevant = len(relevant_docs)
        ranked_docs = self._searcher._ranker.rank_relevant_docs(relevant_docs, query_vector)
        return n_relevant, self._searcher._ranker.retrieve_top_k(ranked_docs, k)

    def search_and_rank_query(self, query, k, p):
        query_as_list = self._parser.parse_sentence(query)[0]
        query_dict, max_tf_query = self.get_query_dict(query_as_list)
        relevant_docs, query_vector = self.relevant_docs_from_posting(query_dict, p)
        ranked_docs = self._ranker.rank_relevant_docs(relevant_docs, query_vector)
        return ranked_docs[:k]

    def test_and_clean(self):
        p = 0.0008
        num_of_terms = round(p * len(self._indexer.inverted_idx_term))
        sorted_index = sorted(self._indexer.inverted_idx_term.items(), key=lambda item: item[1][0], reverse=True)

        for i in range(num_of_terms):
            print(sorted_index[i][0])
            del self._indexer.inverted_idx_term[sorted_index[i][0]]

        for term in list(self._indexer.inverted_idx_term.keys()):
            # TODO - make statistics
            if self._indexer.inverted_idx_term[term][0] <= 1:
                del self._indexer.inverted_idx_term[term]

    def write_to_csv(tuple_list):
        df = pd.DataFrame(tuple_list, columns=['query', 'tweet_id', 'score'])
        df.to_csv('results.csv')
