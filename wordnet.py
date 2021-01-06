from nltk.corpus import wordnet


class Wordnet:

    def __init__(self):
        pass

    # def get_term_synonym(self, term):
    #     term_synonyms = list()
    #     for synset in wordnet.synsets(term):
    #         for lemma in synset.lemmas():
    #             # first_word = wordnet.synset("Travel.v.01")
    #             first_word = wordnet.synset(term)
    #             sim = first_word.wup_similarity(lemma)
    #             if sim > 0.7:
    #                 term_synonyms.append(lemma.name())
    #     print(term_synonyms)
    #     return term_synonyms

    def get_term_synonym(self, term):
        synonym = None
        try:
            synset_lemmas = wordnet.synsets(term)[0].lemmas()
            # synset_words= [lemma.name() for lemma in synset_lemmas]
            # print(synset_words)

            # if len(synset_lemmas) > 2:
            lemma = synset_lemmas[0].name()
            if lemma.lower() != term:
                synonym = lemma
                # print(synonym)
                # print('#################')
        except:
            return synonym
        return synonym

    # expand original query dict and return expanded query dict
    def expand_query(self, query_dict, max_tf_query):
        query_terms = list(query_dict.keys())
        expanded_query_dict = query_dict
        # all_synonym = set()
        for term in query_terms:
            term_syn = self.get_term_synonym(term.lower())

            if term_syn and term_syn not in expanded_query_dict:
                expanded_query_dict[term_syn] = 1.0/max_tf_query
        return expanded_query_dict
