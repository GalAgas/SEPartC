import pandas as pd
from reader import ReadFile
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
from ranker import Ranker
from localMethod import LocalMethod
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
        self._indexer.save_index("inverted_idx")
        # TODO - check the name of inverted_idx
        # self._indexer.save_index("idx_bench")

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
        self._indexer.entities_and_small_big()
        ###########
        self.test_and_clean()
        ###########
        self._indexer.calculate_idf(self._parser.number_of_documents)
        self._indexer.save_index("idx_bench")
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

########################################################################################################################
    def get_query_dict(self, tokenized_query):
        max_tf = 1
        query_dict = {}
        for index, term in enumerate(tokenized_query):
            if term not in query_dict:
                query_dict[term] = 1

            else:
                query_dict[term] += 1
                if query_dict[term] > max_tf:
                    max_tf = query_dict[term]

        for term in query_dict:
            query_dict[term] /= max_tf

        return query_dict, max_tf

    def relevant_docs_from_posting(self, query_dict):
        relevant_docs = {}
        query_vector = np.zeros(len(query_dict), dtype=float)

        # TODO - check after new parser
        # p_threshold = 0.2
        # # full_cells_threshold = math.ceil(p_threshold * len(query_vector))
        # full_cells_threshold = round(p_threshold * len(query_vector))

        for idx, term in enumerate(list(query_dict.keys())):
            try:
                tweets_per_term = self._indexer.get_term_posting_tweets_dict(term)
                # if tweets_per_term is None:
                #     print(term)
                for tweet_id, vals in tweets_per_term.items():
                    if tweet_id not in relevant_docs.keys():
                        relevant_docs[tweet_id] = np.zeros(len(query_dict), dtype=float)

                    # Wij - update tweet vector in index of term with tf-idf
                    tf_tweet = vals[0]
                    idf_term = self._indexer.get_term_idf(term)
                    relevant_docs[tweet_id][idx] = tf_tweet * idf_term

                    # Wiq - update query vector in index of term with tf-idf
                    tf_query = query_dict[term]
                    query_vector[idx] = tf_query * idf_term
            except:
                pass
        return relevant_docs, query_vector

########################################################################################################################

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
        searcher = Searcher(self._parser, self._indexer, model=self._model)
        searcher.set_method_type('3')

        round_1 = self.search_and_rank_query(query, 100)
        local = LocalMethod(self._indexer)
        expanded_query = local.expand_query(query, round_1)
        round_2 = self.search_and_rank_query(expanded_query, None)
        # n_relevant, ranked_doc_ids
        round_2 = [seq[0] for seq in round_2]
        return len(round_2), round_2


    def search_and_rank_query(self, query, k):
        query_as_list = self._parser.parse_sentence(query)[0]
        query_dict, max_tf_query = self.get_query_dict(query_as_list)
        relevant_docs, query_vector = self.relevant_docs_from_posting(query_dict)
        ranked_docs = self._ranker.rank_relevant_docs(relevant_docs, query_vector)
        return ranked_docs[:k]

    def test_and_clean(self):
        for term in list(self._indexer.inverted_idx_term.keys()):
            # TODO - make statistics
            if len(self._indexer.inverted_idx_term[term][0]) <= 1:
                del self._indexer.inverted_idx_term[term]

    def write_to_csv(tuple_list):
        df = pd.DataFrame(tuple_list, columns=['query', 'tweet_id', 'score'])
        df.to_csv('results.csv')
