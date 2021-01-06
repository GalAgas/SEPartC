import math
from collections import Counter
import utils


# DO NOT MODIFY CLASS NAME
class Indexer:
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def __init__(self, config):
        # STRUCTURE OF INDEX
        # inverted_idx - {term: [df, {tweet_id:[norm_tf,tf, max_tf?, doc_len]}, ,idf]}

        self.inverted_idx = {}
        self.posting_dict = {}
        self.config = config
        self.entities = Counter()
        self.small_big = {}


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
        # entities as is ->  token
        self.entities.update(document.entities_set)

        # Go over each term in the doc
        for term in document_dictionary.keys():
            try:
                term_low = term.lower()
                # small_big
                # term = lower case
                if term_low in document.small_big_letters_dict:
                    if term_low not in self.small_big.keys():
                        self.small_big[term_low] = document.small_big_letters_dict[term_low]
                    else:
                        # self.small_big[term] = self.small_big[term] and document.small_big_letters_dict[term]
                        self.small_big[term_low] = self.small_big[term_low] or document.small_big_letters_dict[term_low]

                # Update inverted index and posting
                if term not in self.inverted_idx.keys():
                    self.inverted_idx[term] = [1, {}]
                else:
                    self.inverted_idx[term][0] += 1

                tf = document_dictionary[term]
                normalized_tf = tf/document.max_tf
                self.inverted_idx[term][1][document.tweet_id] = [normalized_tf, tf]

            except:
                pass

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        # return utils.load_obj(self.config.get_savedFileMainFolder() + "\\" + fn)
        return utils.load_obj(self.config.get_savedFileMainFolder() + fn)

    # TODO - fix this for tests!!
    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def save_index(self, fn):
        """
        Saves a pre-computed index (or indices) so we can save our work.
        Input:
              fn - file name of pickled index.
        """
        # utils.save_obj(self.inverted_idx, self.config.get_savedFileMainFolder() + "\\" + fn)
        # TODO - fix this for tests!!
        utils.save_obj(self.inverted_idx, fn)

    # determine the final form of the saved term in inverted_index
    # small or big letter, save or del entity
    def entities_and_small_big(self):
        for term in list(self.inverted_idx.keys()):
            # bad entity
            if term in self.entities and self.entities[term] < 2:
                del self.inverted_idx[term]

            # only big letters
            if term in self.small_big and not self.small_big[term]:
                lower_term = term
                term = term.upper()
                self.inverted_idx[term] = self.inverted_idx[lower_term]
                del self.inverted_idx[lower_term]

    # Calculate idf for each term in inverted index after finish indexing
    def calculate_idf(self, N):
        for val in self.inverted_idx.values():
            val.append(math.log2(N/val[0]))

    def get_term_posting_tweets_dict(self, term):
        if term and term in self.inverted_idx:
            return self.inverted_idx[term][1]

    def get_term_idf(self, term):
        if term and term in self.inverted_idx:
            return self.inverted_idx[term][-1]


