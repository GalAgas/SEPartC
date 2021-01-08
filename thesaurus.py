from nltk.corpus import lin_thesaurus as thesaurus
from nltk import pos_tag


class Thesaurus:

    def __init__(self, searcher):
        self.searcher = searcher
        self.p_threshold = 0.2
        # for best
        self.expanded_query_dict = None
        self.max_tf_query = None

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

            print(f"thesaurus expand_query -> {term_syn}")

            if term_syn and term_syn not in expanded_query_dict and self.searcher.is_term_in_index(term_syn):
                expanded_query_dict[term_syn] = 1.0/max_tf_query
        self.expanded_query_dict = expanded_query_dict
        self.max_tf_query = max_tf_query
        return expanded_query_dict
