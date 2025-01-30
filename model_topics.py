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


class TopicModeler(object):
    """
    Performs all the topic modeling stuff

    Model Steps:
    1.
    """
    def __init__(self):
        self.doc_paths: iter = None

        # todo: extend stopwords for eg tbls, cup, oz, etc
        self.stop_words = set(stopwords.words('english'))
        # would prefer to use a lemmatizer
        # todo: work to pos-tag sentences so can lemmatize
        self.stemmer = PorterStemmer()

        self.bigram_mod = None
        self.trigram_mod = None

        # the dictionary
        self.id2word: gensim.corpora.Dictionary = None
        # the word frequency matrix with docs on rows, words on columns
        self.corpus: iter = None

        # models
        self.tfidf: gensim.models.TfidfModel = None

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
        ngram_sentence = self.trigram_mod[
            self.bigram_mod[
                self.sentence_to_words(sentence)
            ]
        ]
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

    def build_dict_and_corpus(self):
        self.id2word = Dictionary(self.yield_clean_docs())
        # in one case changing no_above from 0.9 to 0.6 filtered only about 5 words
        # no_below seems to have a greater effect 3->5 resulted in 3411->2648
        self.id2word.filter_extremes(no_below=8, no_above=0.6)
        # self.id2word.filter_extremes(no_below=4, no_above=0.9)  # testing

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
    test_paths = [
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
    # test_paths = []
    # with os.scandir('/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/') as it:
    #     for entry in it:
    #         if entry.name.endswith('.md') and entry.is_file() and not entry.name.startswith('.'):
    #             test_paths.append(str(entry.path))

    tm = TopicModeler()
    tm.doc_paths = test_paths

    # yield_doc_text
    # for txt in tm.yield_doc_text():
    #     print(txt, end='--------------\n\n')

    # yield_doc_sentences
    # for sentences in tm.yield_doc_sentences():
    #     print(sentences, end='--------------\n\n')

    # sentence_to_words
    # for sentences in tm.yield_doc_sentences():
    #     for sentence in sentences:
    #         print(f'sentence: {sentence}')
    #         print(f'words: {tm.sentence_to_words(sentence)}', end='\n---------------------------\n')

    # yield_all_sentences
    # for sentence in tm.yield_all_sentences():
    #     print(sentence)

    # build ngrams
    # tm.build_ngrams()
    # for sentence in tm.yield_all_sentences():
    #     for word in tm.bigram_mod[sentence]:
    #         if '_' in word:
    #             print(word)

    # clean_sentence
    # tm.build_ngrams()
    # for sentences in tm.yield_doc_sentences():
    #     for sentence in sentences:
    #         print(f'sentence: {sentence}')
    #         print(f'cleaned: {tm.clean_sentence(sentence)}')
    #         print('---------------------')

    # yield_clean_docs
    # tm.build_ngrams()
    # for doc in tm.yield_clean_docs():
    #     print(doc, end='\n\n')

    # build_dictionary
    tm.build_ngrams()
    tm.build_dict_and_corpus()
    print(f'len(dictionary) = {len(tm.id2word)}')
    print(f'10 most common dict words: {tm.id2word.most_common(10)}')
    # for doc in tm.yield_clean_docs():
    #     print(tm.id2word.doc2bow(doc))

    # tfidf
    tm.build_tfidf()
    # for doc in tm.yield_corpus():
    #     print(tm.tfidf[doc])

    """
    # compare with lsi
    model_lsi = tm.return_lsi_model(num_topics=40)
    for top in model_lsi.print_topics():
        print(top)
    index = similarities.MatrixSimilarity(model_lsi[[bow for bow in tm.yield_corpus()]])
    with open("/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Honey-Mustard-Salmon.md", 'r') as f:
        doc = f.read()
    vec_bow = tm.id2word.doc2bow(tm.clean_doc(doc))
    vec_lsi = model_lsi[vec_bow]
    # similar recipes
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for doc_pos, doc_score in sims[:10]:
        print(doc_score, test_paths[doc_pos].split('/')[-1])
    for doc_pos, doc_score in sims[-10:]:
        print(doc_score, test_paths[doc_pos].split('/')[-1])
    """

    # model_lda = tm.return_lda_model(num_topics=20)
    # for top in model_lda.print_topics(10):
    #     print(top)
    # pyLDAvis.enable_notebook()
    # vis = pyLDAvis.gensim.prepare(model_lda, corpus=list(tm.yield_corpus()), dictionary=tm.id2word)
    # pyLDAvis.save_html(vis, 'test_vis_01.html')
    

if __name__ == '__main__':
    tests()
