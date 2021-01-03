import search_engine_best
from search_engine_1 import SearchEngine
from configuration import ConfigClass

if __name__ == '__main__':
    corpus_path = 'C:\\Gal\\University\\Third_year\\semA\\InformationRetrieval\\SearchEngineProject\\Data\\Data\\date=07-27-2020'
    output_path = 'out'
    stemming = False
    # queries = ['Dr. Anthony Fauci wrote in a 2005 paper published in Virology Journal that hydroxychloroquine was effective in treating SARS.']
    queries = 'queries.txt'
    num_docs_to_retrieve = 20
    config = ConfigClass()
    se_1 = SearchEngine(config)
    se_1.main_method(corpus_path, output_path, stemming, queries, num_docs_to_retrieve)


    # search_engine_best.main()
