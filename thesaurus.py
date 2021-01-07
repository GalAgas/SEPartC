from nltk.corpus import lin_thesaurus as thesaurus
from nltk import pos_tag


class Thesaurus:

    def __init__(self):
        pass

    # # max score
    # def get_term_synonym(self, term):
    #     chosen_syn = None
    #     max_score = 0
    #     try:
    #         syn_types = thesaurus.scored_synonyms(term)
    #         for type in syn_types:
    #             if len(type[1]) > 0:
    #                 syn_score_tup = list(type[1])[0]
    #                 curr_syn = syn_score_tup[0]
    #                 curr_score = syn_score_tup[1]
    #                 if curr_score > max_score:
    #                     max_score = curr_score
    #                     chosen_syn = curr_syn
    #     except:
    #         return chosen_syn
    #     return chosen_syn
    #
    # # expand original query dict and return expanded query dict
    # def expand_query(self, query_dict, max_tf_query):
    #     query_terms = list(query_dict.keys())
    #     expanded_query_dict = query_dict
    #     for term in query_terms:
    #         term_syn = self.get_term_synonym(term.lower())
    #
    #         if term_syn and term_syn not in expanded_query_dict:
    #             expanded_query_dict[term_syn] = 1.0/max_tf_query
    #     return expanded_query_dict

    # tag of speech
    def get_term_synonym(self, tagged_term):
        chosen_syn = None
        try:
            syn_types = thesaurus.synonyms(tagged_term[0])
            part_of_speech = tagged_term[1]
            if part_of_speech.startswith('V'):
                type = syn_types[2]
            elif part_of_speech.startswith('J'):
                type = syn_types[0]
            else:
                type = syn_types[1]

            if len(type[1]) > 0:
                chosen_syn = list(type[1])[0]

        except:
            return chosen_syn
        return chosen_syn

    # expand original query dict and return expanded query dict
    def expand_query(self, query_dict, max_tf_query):
        query_terms = list(query_dict.keys())
        tagged_query_terms = pos_tag(query_terms)
        expanded_query_dict = query_dict
        for tagged_term in tagged_query_terms :
            term_syn = self.get_term_synonym(tagged_term)

            if term_syn and term_syn not in expanded_query_dict:
                expanded_query_dict[term_syn] = 1.0/max_tf_query
        return expanded_query_dict
