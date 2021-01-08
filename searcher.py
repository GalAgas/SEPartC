from ranker import Ranker
import numpy as np
from wordnet import Wordnet
from thesaurus import Thesaurus
from localMethod import LocalMethod
from spellChecker import MySpellCheker

# DO NOT MODIFY CLASS NAME
class Searcher:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit. The model 
    # parameter allows you to pass in a precomputed model that is already in 
    # memory for the searcher to use such as LSI, LDA, Word2vec models. 
    # MAKE SURE YOU DON'T LOAD A MODEL INTO MEMORY HERE AS THIS IS RUN AT QUERY TIME.
    def __init__(self, parser, indexer, model=None):
        self._parser = parser
        self._indexer = indexer

        self._ranker = Ranker()
        self._model = model
        self._config = self._indexer.config
        self._method_class = None

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def search(self, query, k=None):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results (tweet ids).
        Input:
            query - string.
            k - number of top results to return, default to everything.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        self._indexer.load_index("idx_bench.pkl")
        query_as_list = self._parser.parse_sentence(query)[0]
        query_dict, max_tf_query = self.get_query_dict(query_as_list)
        expanded_query_dict = self._method_class.expand_query(query_dict, max_tf_query)
        return self.search_helper(expanded_query_dict, k, self._method_class.p_threshold, self._method_class.p_rel)

    # create {term : tf} for query
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

    def relevant_docs_from_posting(self, query_dict, p_threshold=0):
        relevant_docs = {}
        query_vector = np.zeros(len(query_dict), dtype=float)
        full_cells_threshold = round(p_threshold * len(query_vector))

        for idx, term in enumerate(list(query_dict.keys())):
            try:
                # added
                docs_index = self.get_doc_index()
                tweets_per_term = self._indexer.get_term_posting_tweets_dict(term)

                for tweet_id, vals in tweets_per_term.items():
                    doc_date = docs_index[tweet_id][1]
                    if tweet_id not in relevant_docs.keys():
                        relevant_docs[tweet_id] = [np.zeros(len(query_dict), dtype=float),doc_date]

                    # Wij - update tweet vector in index of term with tf-idf
                    tf_tweet = vals[0]
                    idf_term = self._indexer.get_term_idf(term)
                    relevant_docs[tweet_id][0][idx] = tf_tweet * idf_term

                    # Wiq - update query vector in index of term with tf-idf
                    tf_query = query_dict[term]
                    query_vector[idx] = tf_query * idf_term
            except:
                pass

        # OPTIMIZATIONS
        for doc in list(relevant_docs.keys()):
            if np.count_nonzero(relevant_docs[doc][0]) < full_cells_threshold:
                del relevant_docs[doc]

        return relevant_docs, query_vector

    def set_method_type(self, method_type):
        if method_type == '1':
            self._method_class = Wordnet(self)
        elif method_type == '2':
            self._method_class = Thesaurus(self)
        elif method_type == '3':
            self._method_class = LocalMethod(self)
        elif method_type == '4':
            self._method_class = MySpellCheker(self)
        # elif.. more methods


    def get_term_index(self):
        return self._indexer.inverted_idx_term

    def get_doc_index(self):
        return self._indexer.inverted_idx_doc

    def is_term_in_index(self, term):
        return term in self._indexer.inverted_idx_term

    def search_helper(self, query_dict, k, p_threshold=0, p_relevant=0):
        relevant_docs, query_vector = self.relevant_docs_from_posting(query_dict, p_threshold)
        n_relevant = len(relevant_docs)
        ranked_docs = self._ranker.rank_relevant_docs(relevant_docs, query_vector)
        return n_relevant, self._ranker.retrieve_top_k(ranked_docs, k, p_relevant)