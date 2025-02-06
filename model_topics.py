import os
from collections.abc import Iterator

import gensim.corpora
from gensim import similarities
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
from gensim.models import TfidfModel, LsiModel, LdaModel, CoherenceModel

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
    Provides memory efficient methods for cleaning documents
    doc_paths: iter of paths to docs
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
        # self.id2word.filter_extremes(no_below=8, no_above=0.6)

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
    Yields clean documents from a DocumentPreProcessor object.
    The __len__ method is necessary to build a similarity matrix in memory
    using gensim.similarities.MatrixSimilarity.
    """
    def __init__(self, dpp: DocumentPreProcessor):
        self.dpp: DocumentPreProcessor = dpp

    def __iter__(self) -> Iterator[list[tuple[float]]]:
        for doc in self.dpp.yield_clean_docs():
            yield self.dpp.id2word.doc2bow(doc)

    def __len__(self) -> int:
        return len(self.dpp.doc_paths)


def main():
    doc_paths = []
    dir_path = 'recipes/'
    with os.scandir(dir_path) as it:
        for entry in it:
            if entry.name.endswith('.md') and entry.is_file() and not entry.name.startswith('.'):
                doc_paths.append(str(entry.path))

    dpp = DocumentPreProcessor(doc_paths)

    dpp.extend_stopwords([
        'tsp', 'teaspoon', 'teaspoons',
        'milliliter', 'ml', 'milliliters',
        'tbsp', 'tablespoon', 'tablespoons',
        'cup', 'cups',
        'pound', 'lb', 'pounds',
        'oz', 'ounce', 'ounces',
        'minute', 'minutes',
        'hour', 'hours',
        'high', 'medium', 'low',
    ])

    dpp.build_ngrams()
    dpp.build_dictionary()
    print(f'pre-filter len(dictionary) = {len(dpp.id2word)}')
    dpp.id2word.filter_extremes(no_below=8, no_above=0.5)
    # dpp.save_dict_and_ngrams('test_models')

    # dpp.load_dict_and_ngrams('test_models')
    print(f'final len(dictionary) = {len(dpp.id2word)}')
    print(f'10 most common dict words: {dpp.id2word.most_common(10)}')

    # specific word frequency
    wrd = 'duck'
    wordid = dpp.id2word.token2id[wrd]
    print(f'{wrd} id: {wordid}')
    print(f'{wrd} occures in {dpp.id2word.dfs[wordid]} docs')
    print(f'{wrd} occres {dpp.id2word.cfs[wordid]} times total')

    words_fs_sorted = sorted(dpp.id2word.dfs.items(), key=lambda item: item[1])
    print('10 words in fewest documents')
    for w in words_fs_sorted[:20]:
        print(f'\t{w[1]} x {dpp.id2word[w[0]]}')
    print('10 words in most documents')
    for w in words_fs_sorted[-10:]:
        print(f'\t{w[1]} x {dpp.id2word[w[0]]}')

    # compare using lsi
    corpus = MemoryFriendlyCorpus(dpp)
    tfidf_model = TfidfModel(corpus=corpus)
    tfidf_corpus = tfidf_model[corpus]


    # doc to compare
    compare_path = "test_doc.md"
    with open(compare_path, 'r') as f:
        doc = f.read()
    vec_bow = dpp.id2word.doc2bow(dpp.doc_to_clean_words(doc))

    # comparison
    print(f'---------- LSA ------------')
    model_lsi = LsiModel(
            corpus=tfidf_corpus,
            id2word=dpp.id2word,
            num_topics=15,
        )
    print('lsa topics')
    for top in model_lsi.print_topics():
        print(top)
    index = similarities.MatrixSimilarity(model_lsi[list(tfidf_corpus)])
    vec_lsi = model_lsi[tfidf_model[vec_bow]]
    print(f'vec lsi: {vec_lsi}')
    # similar recipes
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print('10 most similar docs and cosine similarity')
    for doc_pos, doc_score in sims[:10]:
        print(doc_score, doc_paths[doc_pos].split('/')[-1])
    print('10 least similar')
    for doc_pos, doc_score in sims[-10:]:
        print(doc_score, doc_paths[doc_pos].split('/')[-1])

    # lda
    print(f'---------- LDA ------------')
    model_lda = LdaModel(
        corpus=tfidf_corpus,
        num_topics=10,
        id2word=dpp.id2word,
        # random_state=100,
        # update_every=1,
        # chunksize=100,
        passes=5,
        alpha='auto',
        per_word_topics=True,
    )
    print('lda topics')
    for top in model_lda.print_topics():
        print(top)

    # find coherence values for different numbers of topics
    print('finding topic coherence scores')
    coherence_vals = []
    print('topic: ', end='')
    rng = list(range(5, 25, 1))
    for num_topics in rng:
        print(f'{num_topics}, ', end='')
        model_lda = LdaModel(
            corpus=tfidf_corpus,
            num_topics=num_topics,
            id2word=dpp.id2word,
            random_state=100,
            update_every=1,
            chunksize=100,
            passes=2,
            alpha='auto',
        )
        coherencemod = CoherenceModel(model=model_lda, texts=dpp.yield_clean_docs(), coherence='c_v')
        coherence_vals.append(coherencemod.get_coherence())
    print('\ncoherence vals: ')
    for i, co in enumerate(coherence_vals):
        print(f'{rng[i]}: {co}')


if __name__ == '__main__':
    main()
