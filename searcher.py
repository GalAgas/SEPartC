from ranker import Ranker
import numpy as np
from wordnet import Wordnet
from thesaurus import Thesaurus
from localMethod import LocalMethod

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
        # TODO- parse_sentence(query) return :tokenized_text, entities_set, small_big_dict
        # TODO - need to check what to do with entities & small big
        query_as_list = self._parser.parse_sentence(query)[0]
        query_dict, max_tf_query = self.get_query_dict(query_as_list)

        # with wordnet\thesaurus expansion
        #expanded_query_dict = self._method_class.expand_query(query_dict, max_tf_query)
        #relevant_docs, query_vector = self.relevant_docs_from_posting(expanded_query_dict)

        # without wordnet\thesaurus expansion
        relevant_docs, query_vector = self.relevant_docs_from_posting(query_dict)

        # TODO - fix n_relevant if smallest than k? return k
        n_relevant = len(relevant_docs)
        ranked_docs = self._ranker.rank_relevant_docs(relevant_docs, query_vector)
        ranked_doc_ids = self._ranker.retrieve_top_k(ranked_docs, k)
        return n_relevant, ranked_doc_ids

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

        # TODO - OPTIMIZATIONS
        # for doc in list(relevant_docs.keys()):
        #     if np.count_nonzero(relevant_docs[doc]) < full_cells_threshold:
        #         del relevant_docs[doc]

        return relevant_docs, query_vector

    def set_method_type(self, method_type):
        if method_type == '1':
            self._method_class = Wordnet()
        elif method_type == '2':
            self._method_class = Thesaurus()
        elif method_type == '3':
            self._method_class =  (self._indexer)
        # elif.. more methods
