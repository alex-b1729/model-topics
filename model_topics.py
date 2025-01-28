from gensim.utils import simple_preprocess
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
        self.doc_paths = []

        self.stop_words = set(stopwords.words('english'))
        # would prefer to use a lemmatizer
        # todo: work to pos-tag sentences so can lemmatize
        self.stemmer = PorterStemmer()

        self.bigram_mod = None
        self.trigram_mod = None

    def yield_doc_text(self) -> iter:
        """yields the contents of each file in self.doc_paths"""
        for path in self.doc_paths:
            with open(path, encoding='utf-8', errors='ignore') as f:
                yield f.read()

    def yield_doc_sentences(self) -> iter:
        """
        yields list[str] representing the sentences in each doc
        excluding sentences that are blank when stripped
        """
        for doc in self.yield_doc_text():
            # this splits well on periods but keeps new lines as same sentence
            sentences_rough = sent_tokenize(doc)
            # below I split on \n but don't return empty strings
            yield [
                sentence
                for sentence_part in sentences_rough
                for sentence in sentence_part.split('\n')
                if sentence != ''
            ]

    def sentence_to_words(self, sentence: str) -> list:
        """
        returns list of simple_preprocess words
        """
        return simple_preprocess(sentence, deacc=True)

    def yield_all_sentences(self) -> iter:
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

    def clean_sentence(self, sentence: str) -> list:
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

    def yield_clean_docs(self) -> iter:
        """
        yields for each doc a list of stemmed, non-stop words with ngrams applied

        note that
        list_of_lists = [[1,2,3], [9,8,7]]
        print(
            [
                num
                for sub_list in list_of_lists
                for num in sub_list
            ]
        )
        > [1, 2, 3, 9, 8, 7]
        """
        for doc_sentences in self.yield_doc_sentences():
            yield [
                word
                for sentence in doc_sentences
                for word in self.clean_sentence(sentence)
            ]


def tests():
    test_paths = [
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/20-Minute-Beef-Stroganoff.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Arambasici-(Croatian-Sour-Cabbage-Rolls).md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Backyard-Barbecue-Ribs.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Banana-Bread-I.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Meatloaf-III.md",
        "/Users/abrefeld/ab/Scripts/scrapers/wikirecipes/data/recipes/Cantonese-Roast-Duck.md",
    ]
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


if __name__ == '__main__':
    tests()
