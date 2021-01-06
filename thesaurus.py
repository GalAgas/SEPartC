from nltk.corpus import lin_thesaurus as thesaurus


class Thesaurus:

    def __init__(self):
        pass


    def get_term_synonym(self, term):
        chosen_syn = None
        max_score = 0
        try:
            syn_types = thesaurus.scored_synonyms(term)
            for type in syn_types:
                if len(type[1]) > 0:
                    syn_score_tup = list(type[1])[0]
                    curr_syn = syn_score_tup[0]
                    curr_score = syn_score_tup[1]
                    if curr_score > max_score:
                        max_score = curr_score
                        chosen_syn = curr_syn
        except:
            return chosen_syn
        return chosen_syn

    # expand original query dict and return expanded query dict
    def expand_query(self, query_dict, max_tf_query):
        query_terms = list(query_dict.keys())
        expanded_query_dict = query_dict
        for term in query_terms:
            term_syn = self.get_term_synonym(term.lower())

            if term_syn and term_syn not in expanded_query_dict:
                expanded_query_dict[term_syn] = 1.0/max_tf_query
        return expanded_query_dict
