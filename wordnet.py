from nltk.corpus import wordnet
from nltk import pos_tag


class Wordnet:

    def __init__(self, searcher):
        self.searcher = searcher
        self.p_threshold = 0.25
        self.p_rel = 0.9
        # for best
        self.expanded_query_dict = None
        self.max_tf_query = None

    def get_term_synonym(self, tagged_term):
        synonym = None
        try:
            part_of_speech = tagged_term[1]
            if part_of_speech.startswith('V'):
                type = wordnet.VERB
            elif part_of_speech.startswith('R'):
                type = wordnet.ADV
            elif part_of_speech.startswith('J'):
                type = wordnet.ADJ
            else:
                type = wordnet.NOUN
            synset_lemmas = wordnet.synsets(tagged_term[0], pos=type)[0].lemmas()
            lemma = synset_lemmas[0].name()
            if lemma.lower() != tagged_term[0].lower():
                synonym = lemma
        except:
            return synonym
        return synonym

    # expand original query dict and return expanded query dict
    def expand_query(self, query_dict, max_tf_query):
        query_terms = list(query_dict.keys())
        tagged_query_terms = pos_tag(query_terms)
        expanded_query_dict = query_dict
        for tagged_term in tagged_query_terms :
            term_syn = self.get_term_synonym(tagged_term)

            if term_syn and term_syn not in expanded_query_dict and self.searcher.is_term_in_index(term_syn):
                expanded_query_dict[term_syn] = 1.0/max_tf_query
        # for best - local
        self.expanded_query_dict = expanded_query_dict
        self.max_tf_query = max_tf_query
        return expanded_query_dict