from utils.data import download_dir_if_not_exists

class BM25Retriever:
    def __init__(self):
        from pyserini.search.lucene import LuceneSearcher
        download_dir_if_not_exists('data/pyserini_index/wiki_jsonl')
        self.searcher = LuceneSearcher('data/pyserini_index/wiki_jsonl')

    def __call__(self, query, k=2560):
        hit_list = self.searcher.search(query, k)
        
        ordered_score_did = []
        for hit in hit_list:
            ordered_score_did.append((float(hit.score), hit.docid))
        ordered_score_did.sort(key=lambda x: x[0], reverse=True)
        
        return ordered_score_did
