from ranker import Ranker
import utils
import numpy as np


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

        # added from PartA
        self.loaded_posting = None
        self.loaded_posting_name = None
        self.loaded_doc_name = None
        self.loaded_doc = None

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    # def search(self, query, k=None):
    #     """
    #     Executes a query over an existing index and returns the number of
    #     relevant docs and an ordered list of search results (tweet ids).
    #     Input:
    #         query - string.
    #         k - number of top results to return, default to everything.
    #     Output:
    #         A tuple containing the number of relevant search results, and
    #         a list of tweet_ids where the first element is the most relavant
    #         and the last is the least relevant result.
    #     """
    #     query_as_list = self._parser.parse_sentence(query)
    #
    #     relevant_docs = self._relevant_docs_from_posting(query_as_list)
    #     n_relevant = len(relevant_docs)
    #     ranked_doc_ids = Ranker.rank_relevant_docs(relevant_docs)
    #     return n_relevant, ranked_doc_ids


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
        query_as_list = self._parser.parse_sentence(query)[0]
        query_dict = self.get_query_dict(query_as_list)
        relevant_docs, query_vector = self.relevant_docs_from_posting(query_dict)
        ranked_docs = self._ranker.rank_relevant_docs(relevant_docs, query_vector)
        return self._ranker.retrieve_top_k(ranked_docs, k)


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

        return query_dict


    def relevant_docs_from_posting(self, query_dict):

        posting_query_dict = {}
        for term in query_dict:
            # we have that try because of the entities problem
            try:
                posting_query_dict[term] = self._indexer.final_inverted_idx[term][1]
            except:
                pass
        # sort by value
        posting_query_dict = {k: v for k, v in sorted(posting_query_dict.items(), key=lambda item: item[1])}

        relevant_docs = {}
        query_vector = np.zeros(len(query_dict), dtype=float)

        for idx, item in enumerate(posting_query_dict.items()):
            try:
                term = item[0]
                posting_name = item[1]

                # load suitable posting
                if self.loaded_posting_name is None or self.loaded_posting_name != posting_name:
                    self.loaded_posting = utils.load_obj(self._config.get_savedFileMainFolder() + "\\" + str(posting_name))
                    self.loaded_posting_name = posting_name

                for tup in self.loaded_posting[term]:
                    tweet_id = tup[0]

                    if tweet_id not in relevant_docs.keys():
                        relevant_docs[tweet_id] = np.zeros(len(query_dict), dtype=float)

                    tf_tweet = tup[1]
                    idf = self._indexer.final_inverted_idx[term][-1]
                    relevant_docs[tweet_id][idx] = tf_tweet * idf

                    tf_query = query_dict[term]
                    query_vector[idx] = tf_query * idf
            except:
                pass
        return relevant_docs, query_vector


    # # feel free to change the signature and/or implementation of this function
    # # or drop altogether.
    # def _relevant_docs_from_posting(self, query_as_list):
    #     """
    #     This function loads the posting list and count the amount of relevant documents per term.
    #     :param query_as_list: parsed query tokens
    #     :return: dictionary of relevant documents mapping doc_id to document frequency.
    #     """
    #     relevant_docs = {}
    #     for term in query_as_list:
    #         posting_list = self._indexer.get_term_posting_list(term)
    #         for doc_id, tf in posting_list:
    #             df = relevant_docs.get(doc_id, 0)
    #             relevant_docs[doc_id] = df + 1
    #     return relevant_docs
