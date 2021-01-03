import pandas as pd
from reader import ReadFile
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
import utils


# DO NOT CHANGE THE CLASS NAME
class SearchEngine:

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation, but you must have a parser and an indexer.
    def __init__(self, config=None):
        self._config = config
        # self._parser = Parse()
        self._parser = Parse(self._config)
        self._indexer = Indexer(self._config)
        self._model = None

    def run_engine(self):
        """
        :return:
        """
        r = ReadFile(corpus_path=self._config.get__corpusPath())
        number_of_files = 0

        for i, file in enumerate(r.read_corpus()):
            # Iterate over every document in the file
            number_of_files += 1
            for idx, document in enumerate(file):
                # parse the document
                parsed_document = self._parser.parse_doc(document)
                self._indexer.add_new_doc(parsed_document)
        self._indexer.check_last()
        self._indexer.merge_sort_parallel(3)
        self._indexer.calculate_idf(self._parser.number_of_documents)
        avg_doc_len = self._parser.total_len_docs / self._parser.number_of_documents
        utils.save_obj(avg_doc_len, self._config.get_savedFileMainFolder() + "\\data")

        utils.save_obj(self._indexer.inverted_idx, self._config.get_savedFileMainFolder() + "\\inverted_idx")
        utils.save_obj(self._indexer.docs_inverted, self._config.get_savedFileMainFolder() + "\\docs_inverted")

    def main_method(self, corpus_path, output_path, stemming, queries, num_docs_to_retrieve):

        if num_docs_to_retrieve > 2000:
            num_docs_to_retrieve = 2000

        # update configurations
        self._config.set_corpusPath(corpus_path)
        self._config.set_toStem(stemming)
        self._config.set_savedFileMainFolder(output_path)

        # need to change to build_index_from_parquet(self, fn)
        self.run_engine()
        print("finish run engine!")

        # self._indexer = self.load_index("inverted_idx")
        # inverted_docs = self.load_docs_index()
        # avg_doc_len = utils.load_obj(self._config.get_savedFileMainFolder() + "\\" + "data")

        if type(queries) is list:
            queries_list = queries
        else:
            queries_list = [line.strip() for line in open(queries, encoding="utf8")]

        csv_data = []
        for idx, query in enumerate(queries_list):
            relevant_returned = self.search(query, num_docs_to_retrieve)
            # round_1 = search_and_rank_query(config, querie, inverted_index, inverted_docs, 100, avg_doc_len)
            # local_method_ranker = local_method(config, inverted_docs, inverted_index)
            # expanded_query = local_method_ranker.expand_query(querie, round_1)
            # round_2 = search_and_rank_query(config, expanded_query, inverted_index, inverted_docs, num_docs_to_retrieve,
            #                                 avg_doc_len)

            for doc_tuple in relevant_returned:
                print('tweet id: {}, score (unique common words with query): {}'.format(doc_tuple[0], doc_tuple[1]))

        #     for tup in round_2:
        #         csv_data.append((idx+1, tup[0], tup[1]))
        # write_to_csv(csv_data)


    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def build_index_from_parquet(self, fn):
        """
        Reads parquet file and passes it to the parser, then indexer.
        Input:
            fn - path to parquet file
        Output:
            No output, just modifies the internal _indexer object.
        """
        df = pd.read_parquet(fn, engine="pyarrow")
        documents_list = df.values.tolist()
        # Iterate over every document in the file
        number_of_documents = 0
        for idx, document in enumerate(documents_list):
            # parse the document
            parsed_document = self._parser.parse_doc(document)
            number_of_documents += 1
            # index the document data
            self._indexer.add_new_doc(parsed_document)
        print('Finished parsing and indexing.')

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_index(self, fn):
        """
        Loads a pre-computed index (or indices) so we can answer queries.
        Input:
            fn - file name of pickled index.
        """
        return self._indexer.load_index(fn)

    def load_docs_index(self):
        inverted_docs = utils.load_obj(self.config.get_savedFileMainFolder() + "\\" + "docs_inverted")
        return inverted_docs

    # DO NOT MODIFY THIS SIGNATURE
    # You can change the internal implementation as you see fit.
    def load_precomputed_model(self):
        """
        Loads a pre-computed model (or models) so we can answer queries.
        This is where you would load models like word2vec, LSI, LDA, etc. and
        assign to self._model, which is passed on to the searcher at query time.
        """
        pass

        # DO NOT MODIFY THIS SIGNATURE
        # You can change the internal implementation as you see fit.

    def search(self, query, k=None):
        """
        Executes a query over an existing index and returns the number of
        relevant docs and an ordered list of search results.
        Input:
            query - string.
        Output:
            A tuple containing the number of relevant search results, and
            a list of tweet_ids where the first element is the most relavant
            and the last is the least relevant result.
        """
        searcher = Searcher(self._parser, self._indexer, model=self._model)
        return searcher.search(query, k)

    def write_to_csv(tuple_list):

        df = pd.DataFrame(tuple_list, columns=['query', 'tweet_id', 'score'])
        df.to_csv('results.csv')