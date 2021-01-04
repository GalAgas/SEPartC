from nltk.corpus import wordnet

class wordnet:

    def __init__(self):
        pass

    def get_term_synonym(self, term):
        term_synonyms = list()
        for synset in wordnet.synsets(term):
            for lemma in synset.lemmas():
                # first_word = wordnet.synset("Travel.v.01")
                first_word = wordnet.synset(term)
                sim = first_word.wup_similarity(lemma)
                if sim > 0.7:
                    term_synonyms.append(lemma.name())
        print(term_synonyms)
        return term_synonyms
