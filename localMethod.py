import utils
from collections import defaultdict

class LocalMethod:

    def __init__(self, searcher):
        self.searcher = searcher
        self.p_threshold = 0.2

        self.correlation_matrix = []
        # {wi : {doc1:tf1, doc3:tf3},  wj: {doc2:tf2, doc3:tf3}}
        self.relevant_docs_per_term = {}

    def expand_query(self, query_dict, max_tf_query):
        round_1_len, round_1 = self.searcher.search_helper(query_dict, None, 0.5)
        return self.helper_expand(query_dict, max_tf_query, round_1)

    def helper_expand(self, query_dict, max_tf, round_1):
        expend_query_dict = query_dict
        query_list_keys = list(expend_query_dict.keys())
        all_unique_terms = set()

        for tup in round_1:
            unique_terms = self.searcher.get_doc_index()[tup][0]
            all_unique_terms.update(unique_terms)

        for i, term in enumerate(all_unique_terms):
            # try:
            tweets_contain_term_dict = self.searcher._indexer.get_term_posting_tweets_dict(term)
            if tweets_contain_term_dict is None:
                continue
                # tweets_contain_term_dict = self.searcher.get_term_index()[term][1]
            # except Exception:
            #     continue
            # create term in self.relevant_docs_per_term - {wi : {doc1:tf1, doc3:tf3},  wj: {doc2:tf2, doc3:tf3}}
            self.relevant_docs_per_term[term] = {}
            for tweet_id in round_1:
                if tweet_id in tweets_contain_term_dict:
                    self.relevant_docs_per_term[term][tweet_id] = tweets_contain_term_dict[tweet_id][0]

        all_terms = list(self.relevant_docs_per_term.keys())
        query_indexes = []
        for i in range(len(all_terms)):
            self.correlation_matrix.append([])
            if all_terms[i] in query_list_keys:
                query_indexes.append(i)
            for j in range(len(all_terms)):
                wi = all_terms[i]
                wi_tf_dict = self.relevant_docs_per_term[wi]
                wj = all_terms[j]
                wj_tf_dict = self.relevant_docs_per_term[wj]
                self.correlation_matrix[i].append(self.calculate_Cij(wi_tf_dict, wj_tf_dict))


        for index in query_indexes:
            term_list = self.correlation_matrix[index]
            max_index_1, max_index_2 = self.normaliz(term_list, index)
            term_1 = all_terms[max_index_1]
            term_2 = all_terms[max_index_2]

            if term_1 not in expend_query_dict:
                expend_query_dict[term_1] = 1.0/max_tf
            if term_2 not in expend_query_dict:
                expend_query_dict[term_2] = 1.0/max_tf

        print(expend_query_dict)
        return expend_query_dict

    def normaliz(self, term_list, j):
        norm_before_sort = []
        for i in range(len(term_list)):
            devide_val = float(self.correlation_matrix[i][i]) + float(self.correlation_matrix[j][j]) - float(
                               self.correlation_matrix[i][j])
            if devide_val == 0:
                norm_before_sort.append(0)
            else:
                norm_before_sort.append(float(self.correlation_matrix[i][j]) / float(
                    (self.correlation_matrix[i][i]) + float(self.correlation_matrix[j][j]) - float(
                        self.correlation_matrix[i][j])))

        # remove the query term
        norm_before_sort[j] = 0
        norm = norm_before_sort.copy()
        norm.sort(reverse=True)
        # return norm_before_sort.index(norm[0])
        return norm_before_sort.index(norm[0]), norm_before_sort.index(norm[1])

    def calculate_Cij(self, wi_tf_dict, wj_tf_dict):
        cij = 0
        for doc_id in wi_tf_dict:
            if doc_id in wj_tf_dict:
                cij += wi_tf_dict[doc_id] * wj_tf_dict[doc_id]
        return cij
