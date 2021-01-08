from spellchecker import SpellChecker


class MySpellCheker:

    def __init__(self, searcher):
        self.searcher = searcher
        self.p_threshold = 0.2

    def expand_query(self, query_dict, max_tf_query):
        spell = SpellChecker()
        query_terms = list(query_dict.keys())
        corr_query_dict = query_dict

        for term in query_terms:
            corr = spell.correction(term)
            if term != corr:
                if corr in self.searcher.get_term_index():
                    corr_query_dict[corr] = corr_query_dict[term]
                    del corr_query_dict[term]
        return corr_query_dict
