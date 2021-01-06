from spellchecker import SpellChecker


class MySpellCheker:

    def expand_query(self, query_dict, max_tf_query):
        spell = SpellChecker()
        query_terms = list(query_dict.keys())
        corr_query_dict = query_dict

        for term in query_terms:
            corr = spell.correction(term)
            if term != corr:
                corr_query_dict[term] = corr
        return corr_query_dict


