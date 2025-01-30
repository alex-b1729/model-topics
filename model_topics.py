import os
from collections.abc import Iterator

import gensim.corpora
from gensim import similarities
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models import TfidfModel, LsiModel
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer


# references:
# https://www.datacamp.com/tutorial/discovering-hidden-topics-python
# https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/

# before you run the first time you'll likely need to run this to download nltk stuff
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt_tab')


class DocumentPreProcessor(object):
    """
    Cleans documents and provides memory efficient methods
    doc_paths: iter of paths to docs
    builds dictionary given docs but doesn't filter extremes
    """
    def __init__(self, doc_paths: iter = None):
        self.doc_paths: iter = doc_paths

        self.stop_words = list(stopwords.words('english'))
        # would prefer to use a lemmatizer
        # todo: work to pos-tag sentences so can lemmatize
        self.stemmer = PorterStemmer()

        self.bigram_mod: gensim.models.phrases.Phrases = None
        self.trigram_mod: gensim.models.phrases.Phrases = None

        # the dictionary
        self.id2word: gensim.corpora.Dictionary = None
        # the word frequency matrix with docs on rows, words on columns
        self.corpus: iter = None

        # models
        self.tfidf: gensim.models.TfidfModel = None

    def extend_stopwords(self, sw: list[str]) -> None:
        self.stop_words.extend(sw)

    def yield_doc_text(self) -> Iterator[str]:
        """yields the contents of each file in self.doc_paths"""
        for path in self.doc_paths:
            with open(path, encoding='utf-8', errors='ignore') as f:
                yield f.read()

    def doc_to_sentences(self, doc: str) -> list[str]:
        # this splits well on periods but keeps new lines as same sentence
        sentences_rough = sent_tokenize(doc)
        # below I split on \n but don't return empty strings
        return [
            sentence
            for sentence_part in sentences_rough
            for sentence in sentence_part.split('\n')
            if sentence != ''
        ]

    def yield_doc_sentences(self) -> Iterator[list[str]]:
        """
        yields list[str] representing the sentences in each doc
        excluding sentences that are blank when stripped
        """
        for doc in self.yield_doc_text():
            yield self.doc_to_sentences(doc)

    def sentence_to_words(self, sentence: str) -> list[str]:
        """
        returns list of simple_preprocess words
        """
        return simple_preprocess(sentence, deacc=True)

    def yield_all_sentences(self) -> Iterator[list[str]]:
        """iter all sentences with simple_preprocess applied"""
        for doc in self.yield_doc_sentences():
            for sentence in doc:
                processed_sentence = self.sentence_to_words(sentence)
                if processed_sentence:  # avoids yielding empty lists
                    yield processed_sentence

    def build_bigrams(self) -> None:
        """
        builds the bigram model
        see https://radimrehurek.com/gensim/models/phrases.html
        for ngram implementation in gensim
        """
        self.bigram_mod = Phrases(
            self.yield_all_sentences(),
            min_count=5, threshold=10,  # defaults, but this makes explicit
            connector_words=ENGLISH_CONNECTOR_WORDS,
        )

    def build_trigrams(self) -> None:
        """
        Builds the trigram model
        Must be called after self.build_bigrams()
        """
        self.trigram_mod = Phrases(
            self.bigram_mod[self.yield_all_sentences()],
            min_count=5, threshold=10,  # defaults, but this makes explicit
            connector_words=ENGLISH_CONNECTOR_WORDS,
        )

    def build_ngrams(self) -> None:
        """
        Builds the ngram models
        note that ngrams are built without stemming
        since I think the stem could contain important info for the ngram
        E.g. "asset pricing" is it's own thing and not to be confused with "asset pric"
        but open to be convinced otherwise
        """
        self.build_bigrams()
        self.build_trigrams()

    def sentence_to_clean_words(self, sentence: str) -> list[str]:
        """return list of stemmed, non-stop words, with ngram models applied"""
        ngram_sentence = self.sentence_to_words(sentence)
        if self.bigram_mod:
            ngram_sentence = self.bigram_mod[ngram_sentence]
            if self.trigram_mod:
                ngram_sentence = self.trigram_mod[ngram_sentence]
        return [
            self.stemmer.stem(word)
            for word in ngram_sentence
            if word not in self.stop_words
        ]

    def doc_to_clean_words(self, doc: str) -> list[str]:
        """
        given string document,
        returns list of stemmed, non-stop words with ngrams applied
        """
        return [
            word
            for sentence in self.doc_to_sentences(doc)
            for word in self.sentence_to_clean_words(sentence)
        ]

    def yield_clean_docs(self) -> Iterator[list[str]]:
        """yields lists of clean words for each doc"""
        for doc in self.yield_doc_text():
            yield self.doc_to_clean_words(doc)

    def build_dictionary(self):
        self.id2word = Dictionary(self.yield_clean_docs())
        # in one case changing no_above from 0.9 to 0.6 filtered only about 5 words
        # no_below seems to have a greater effect 3->5 resulted in 3411->2648
        # self.id2word.filter_extremes(no_below=8, no_above=0.6)
        # self.id2word.filter_extremes(no_below=4, no_above=0.9)  # testing

    def save_dict_and_ngrams(self, dir_path: str) -> None:
        """
        saves all of id2word, bigram_mod, trigram_mod that aren't None
        Note this freezes the ngram phrase models!
        """
        if self.id2word:
            self.id2word.save(os.path.join(dir_path, 'id2word.dict'))
        if self.bigram_mod:
            self.bigram_mod.freeze()
            self.bigram_mod.save(os.path.join(dir_path, 'bigram_mod.pkl'))
        if self.trigram_mod:
            self.trigram_mod.freeze()
            self.trigram_mod.save(os.path.join(dir_path, 'trigram_mod.pkl'))

    def load_dict_and_ngrams(self, dir_path: str) -> None:
        """
        loads any of id2word.dict, bigram_mod.pkl, trigram_mod.pkl that exist
        in dir_path directory
        """
        dict_path = os.path.join(dir_path, 'id2word.dict')
        if os.path.exists(dict_path):
            self.id2word = Dictionary.load(dict_path)
        bigram_path = os.path.join(dir_path, 'bigram_mod.pkl')
        if os.path.exists(bigram_path):
            self.bigram_mod = Phrases.load(bigram_path)
        trigram_path = os.path.join(dir_path, 'trigram_mod.pkl')
        if os.path.exists(trigram_path):
            self.trigram_mod = Phrases.load(trigram_path)


class MemoryFriendlyCorpus(object):
    """
    yields clean documents from a DocumentPreProcessor object
    The len is necessary if you want to build a similarity matrix in memory
    using gensim.similarities.MatrixSimilarity
    """
    def __init__(self, dpp: DocumentPreProcessor):
        self.dpp: DocumentPreProcessor = dpp

    def __iter__(self):
        for doc in self.dpp.yield_clean_docs():
            yield self.dpp.id2word.doc2bow(doc)

    def __len__(self):
        return len(self.dpp.doc_paths)


class Asdf(object):
    # similarities.MatrixSimilarity requires len(corpus) to work
    # which is why maybe better to create a corpus class
    def yield_corpus(self) -> Iterator[list[tuple[float]]]:
        for doc in self.yield_clean_docs():
            yield self.id2word.doc2bow(doc)

    def build_tfidf(self):
        self.tfidf = TfidfModel(self.yield_corpus())

    def return_lsi_model(self, num_topics: int) -> gensim.models.LsiModel:
        return LsiModel(
            corpus=self.tfidf[self.yield_corpus()],
            id2word=self.id2word,
            num_topics=num_topics,
        )


def tests():
    # small sample
    doc_paths = [
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/20-Minute-Beef-Stroganoff.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Arambasici-(Croatian-Sour-Cabbage-Rolls).md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Backyard-Barbecue-Ribs.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Banana-Bread-I.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Basic-Omelet.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Beef-Burgers-with-Chipotle-Mayo.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Blueberry-Pancakes.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Cabbage-Rolls-in-Tomato-Sauce-(Holubtsi).md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Calzone.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Chicken-Tikka-Masala.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Corn-Bread-Fish.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Meatloaf-III.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Cantonese-Roast-Duck.md",
    ]
    # all paths
    # doc_paths = []
    # with os.scandir('/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/') as it:
    #     for entry in it:
    #         if entry.name.endswith('.md') and entry.is_file() and not entry.name.startswith('.'):
    #             doc_paths.append(str(entry.path))

    dpp = DocumentPreProcessor(doc_paths)
    extra_stopwords = [
        'minute', 'hour', 'oz', 'cup', 'ml', 'tsp', 'tbsp'
    ]
    dpp.extend_stopwords(extra_stopwords)

    # yield_doc_text
    # for txt in dpp.yield_doc_text():
    #     print(txt, end='--------------\n\n')

    # yield_doc_sentences
    # for sentences in dpp.yield_doc_sentences():
    #     print(sentences, end='--------------\n\n')

    # sentence_to_words
    # for sentences in dpp.yield_doc_sentences():
    #     for sentence in sentences:
    #         print(f'sentence: {sentence}')
    #         print(f'words: {dpp.sentence_to_words(sentence)}', end='\n---------------------------\n')

    # yield_all_sentences
    # for sentence in dpp.yield_all_sentences():
    #     print(sentence)

    # build ngrams
    # dpp.build_ngrams()
    # for sentence in dpp.yield_all_sentences():
    #     for word in dpp.bigram_mod[sentence]:
    #         if '_' in word:
    #             print(word)

    # clean_sentence
    # dpp.build_ngrams()
    # for sentences in dpp.yield_doc_sentences():
    #     for sentence in sentences:
    #         print(f'sentence: {sentence}')
    #         print(f'cleaned: {dpp.clean_sentence(sentence)}')
    #         print('---------------------')

    # yield_clean_docs
    # dpp.build_ngrams()
    # for doc in dpp.yield_clean_docs():
    #     print(doc, end='\n\n')

    # build_dictionary
    dpp.build_ngrams()
    dpp.build_dictionary()
    print(f'len(dictionary) = {len(dpp.id2word)}')
    print(f'10 most common dict words: {dpp.id2word.most_common(10)}')
    # for doc in tm.yield_clean_docs():
    #     print(tm.id2word.doc2bow(doc))
    # for doc in tm.yield_clean_docs():
    #     print(doc)

    # tfidf
    # tm.build_tfidf()
    # for doc in tm.yield_corpus():
    #     print(tm.tfidf[doc])

    # compare using lsi
    # model_lsi = tm.return_lsi_model(num_topics=40)
    # for top in model_lsi.print_topics():
    #     print(top)
    # index = similarities.MatrixSimilarity(model_lsi[[bow for bow in tm.yield_corpus()]])
    # with open("/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Honey-Mustard-Salmon.md", 'r') as f:
    #     doc = f.read()
    # vec_bow = tm.id2word.doc2bow(tm.doc_to_clean_words(doc))
    # vec_lsi = model_lsi[vec_bow]
    # # similar recipes
    # sims = index[vec_lsi]
    # sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # for doc_pos, doc_score in sims[:10]:
    #     print(doc_score, test_paths[doc_pos].split('/')[-1])
    # for doc_pos, doc_score in sims[-10:]:
    #     print(doc_score, test_paths[doc_pos].split('/')[-1])

    # model_lda = tm.return_lda_model(num_topics=20)
    # for top in model_lda.print_topics(10):
    #     print(top)
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim.prepare(model_lda, corpus=list(tm.yield_corpus()), dictionary=tm.id2word)
    # pyLDAvis.save_html(vis, 'test_vis_01.html')


def main():
    pass


if __name__ == '__main__':
    tests()
    # main()
