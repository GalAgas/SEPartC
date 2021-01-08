import math
from collections import Counter
import utils
import bisect


# DO NOT MODIFY CLASS NAME
class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        # STRUCTURE OF INDEX
        # inverted_idx - {term: [df, {tweet_id:[norm_tf,tf, max_tf?, doc_len]}, ,idf]}


        # inverted_idx_term - {term:[df, {document.tweet_id: [normalized_tf, tf]}, idf]}
        # tweets_posting - {tweet_id : [document.unique_terms, document.tweet_date_obj, document.max_tf, document.doc_length]}


        self.inverted_idx_term = {}
        self.inverted_idx_doc = {}

        self.config = config
        # self.entities = Counter()
        # self.small_big = {}


    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        :return: -
        """
        document_dictionary = document.term_doc_dictionary
        self.inverted_idx_doc[document.tweet_id] = [document.unique_terms, document.tweet_date_obj, document.max_tf,
                                                document.doc_length]

        # entities as is ->  token
        # self.entities.update(document.entities_set)

        # Go over each term in the doc
        for term in document_dictionary.keys():
            term_low = term.lower()
            # small_big
            # term = lower case
            # if term_low in document.small_big_letters_dict:
            #     if term_low not in self.small_big.keys():
            #         self.small_big[term_low] = document.small_big_letters_dict[term_low]
            #     else:
            #         self.small_big[term_low] = self.small_big[term_low] or document.small_big_letters_dict[term_low]

            # Update inverted index and posting
            if term not in self.inverted_idx_term.keys():
                self.inverted_idx_term[term] = [1, {}]
            else:
                self.inverted_idx_term[term][0] += 1

            tf = document_dictionary[term]
            normalized_tf = tf/document.max_tf

            self.inverted_idx_term[term][1][document.tweet_id] = [normalized_tf, tf]
            self.inverted_idx_term[term][1][document.tweet_id] = [normalized_tf, tf]


    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        index_tup = utils.load_obj(fn)
        self.inverted_idx_doc = index_tup[0]
        self.inverted_idx_term = index_tup[1]

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        index_tup = (self.inverted_idx_doc, self.inverted_idx_term)
        utils.save_obj(index_tup, fn)

    # Calculate idf for each term in inverted index after finish indexing
    def calculate_idf(self, N):
        for val in self.inverted_idx_term.values():
            val.append(math.log2(N/val[0]))

    def get_term_posting_tweets_dict(self, term):
        if term and term in self.inverted_idx_term:
            return self.inverted_idx_term[term][1]

    def get_term_idf(self, term):
        if term and term in self.inverted_idx_term:
            return self.inverted_idx_term[term][-1]


