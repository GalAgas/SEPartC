import utils
from collections import defaultdict

class LocalMethod:

    def __init__(self, indexer):
        self.inverted_docs = indexer.inverted_idx_doc
        self.inverted_term = indexer.inverted_idx_term

        self.correlation_matrix = []
        # {wi : {doc1:tf1, doc3:tf3},  wj: {doc2:tf2, doc3:tf3}}
        self.relevant_docs_per_term = {}

    def expand_query(self, query, round_1):
        query_set = set()
        query_set.update(query.split(' '))

        all_unique_terms = set()
        relevent_tweets_id = {}
        for tup in round_1:
            unique_terms = self.inverted_docs[tup[0]][0]
            all_unique_terms.update(unique_terms)


        for i, term in enumerate(all_unique_terms):
            # TODO - think how to fix (when we have bad entity)
            try:
                tweets_contain_term_dict = self.inverted_term[term][0]
            except Exception:
                continue
            # create term in self.relevant_docs_per_term - {wi : {doc1:tf1, doc3:tf3},  wj: {doc2:tf2, doc3:tf3}}
            self.relevant_docs_per_term[term] = {}
            for tweet_tuple in round_1:
                tweet_id = tweet_tuple[0]
                if tweet_id in tweets_contain_term_dict:
                    self.relevant_docs_per_term[term][tweet_id] = tweets_contain_term_dict[tweet_id][0]

        all_terms = list(self.relevant_docs_per_term.keys())
        query_indexes = []
        query = query
        for i in range(len(all_terms)):
            self.correlation_matrix.append([])
            if all_terms[i] in query:
                query_indexes.append(i)
            for j in range(len(all_terms)):
                wi = all_terms[i]
                wi_tf_dict = self.relevant_docs_per_term[wi]
                wj = all_terms[j]
                wj_tf_dict = self.relevant_docs_per_term[wj]
                self.correlation_matrix[i].append(self.calculate_Cij(wi_tf_dict, wj_tf_dict))


        for index in query_indexes:
            term_list = self.correlation_matrix[index]
            max_index = self.normaliz(term_list, index)
            query_set.add(all_terms[max_index])

        query = ' '.join(str(e) for e in query_set)
        return query

    def normaliz(self, term_list, j):
        norm = []
        for i in range(len(term_list)):
            devide_val = float(self.correlation_matrix[i][i]) + float(self.correlation_matrix[j][j]) - float(
                               self.correlation_matrix[i][j])
            if devide_val == 0:
                norm.append(0)
            else:
                norm.append(float(self.correlation_matrix[i][j]) / float(
                    (self.correlation_matrix[i][i]) + float(self.correlation_matrix[j][j]) - float(
                        self.correlation_matrix[i][j])))

        # remove the query term
        norm[j] = 0
        max_value = max(norm)
        return norm.index(max_value)

    def calculate_Cij(self, wi_tf_dict, wj_tf_dict):
        cij = 0
        for doc_id in wi_tf_dict:
            if doc_id in wj_tf_dict:
                cij += wi_tf_dict[doc_id] * wj_tf_dict[doc_id]
        return cij
