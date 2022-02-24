import operator

from collections import defaultdict
from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import wordnet, stopwords
from syntok.tokenizer import Tokenizer


def get_word_ngrams(text, min_length=1, max_length=4, split_tokens=(',', '.')):
    """
    Returns word ngrams between min_length and max_length tokens.
    :param text: the text to extract word ngrams from
    :param min_length: the minimum number of tokens
    :param max_length: the maximum number of tokens
    :param split_tokens:
    :return: list of word ngrams
    """
    tokens = []
    for index, token in enumerate(text):
        if token in split_tokens:
            token = "#"
        tokens.append((token, index))

    max_length = min(max_length, len(tokens))
    all_ngrams = []
    for n in range(min_length, max_length + 1):
        ngrams = zip(*[tokens[i:] for i in range(n)])
        for ngram in ngrams:
            all_ngrams.append(ngram)

    word_ngrams = [(" ".join([token[0] for token in ngram]), ngram[0][1], ngram[-1][1]) for ngram in all_ngrams]
    word_ngrams = [ngram for ngram in word_ngrams if '#' not in ngram[0]]
    return word_ngrams


def html_to_text(raw_html):
    return BeautifulSoup(raw_html, "lxml").text


def common_words(t1, t2):
    t1_tokens = set(t1.split(" "))
    t2_tokens = set(t2.split(" "))
    return len(t1_tokens.intersection(t2_tokens)) / len(t2_tokens)


class UnsupervisedFacetExtractor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = Tokenizer()
        self.stopwords = set(stopwords.words('english'))

    def _lemmatize(self, text):
        """
        Lemmatizes the tokens of a given text
        :param text:
        :return: lemmatized text
        """
        tokens = [t.value for t in self.tokenizer.tokenize(text)]
        lemmatized_words = []
        try:
            pos_labels = pos_tag(tokens)
            pos_labels = [pos[1][0].lower() for pos in pos_labels]
            for i, word in enumerate(tokens):
                if pos_labels[i] == 'j':
                    pos_labels[i] = 'a'  # 'j' <--> 'a' reassignment
                if pos_labels[i] in ['r']:  # For adverbs it's a bit different
                    try:
                        lemma = wordnet.synset(word + '.r.1').lemmas()[0].pertainyms()[0].name()
                    except:
                        lemma = word
                    lemmatized_words.append(lemma)
                elif pos_labels[i] in ['a', 's', 'v']:  # For adjectives and verbs
                    lemmatized_words.append(self.lemmatizer.lemmatize(word, pos=pos_labels[i]))
                else:  # For nouns and everything else as it is the default kwarg
                    lemmatized_words.append(self.lemmatizer.lemmatize(word))
        except:
            pass

        lemmatized_text = " ".join(lemmatized_words)

        return lemmatized_text

    def valid_ngram(self, pos_tags, i, j):
        if len(pos_tags) == 0:
            return False

        if pos_tags[-1][1] in ["TO", "IN", "DT", "CC", "VBZ", "VBP", "VB", "VBD", ".", ":", "JJ", "MD", "WDT", "PRP$"] or \
                pos_tags[0][1] in ["TO", "IN", "DT", "CC", "VBZ", "VBP", ".", ":"]:
            return False

        tokens = [p[0] for p in pos_tags]
        if not "".join(tokens).isalpha():
            return False

        if len(pos_tags) < 2:
            return False

        return True

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = [t.value for t in tokens]
        return tokens

    def extract(self, batch_queries, batch_snippets, limit=100):
        is_batch = True
        if not isinstance(batch_queries, list):
            batch_queries = [batch_queries]
            batch_snippets = [batch_snippets]
            is_batch = False

        batch_facets = []

        for i, query in enumerate(batch_queries):
            texts = batch_snippets[i]
            query_tokens = self.tokenizer.tokenize(query.lower())
            query_tokens = [q.value for q in query_tokens]
            query = " ".join(query_tokens)
            texts = [self.tokenize(html_to_text(t).lower()) for t in texts]

            tf = defaultdict(lambda: 0.0)
            for text in texts:
                pos_tags = pos_tag(text)
                ngrams = get_word_ngrams(text, min_length=1, max_length=5)
                ngrams = [ngram for ngram in ngrams if ngram[0] not in self.stopwords and len(ngram[0]) > 1 and
                          self.valid_ngram(pos_tags[ngram[1]:ngram[2] + 1], ngram[1], ngram[2] + 1) and ngram[
                              0] not in query]
                for ngram in ngrams:
                    tf[ngram[0].strip()] += 1

            for term, freq in tf.items():
                tf[term] = (1 + 10 * common_words(term, query)) * tf[term]

            tf_sorted = sorted(tf.items(), key=operator.itemgetter(1), reverse=True)

            results = [term[0] for term in tf_sorted[:limit]]

            filtered_results = []
            checked_ngrams = set()
            checked_ngrams.add(query)
            for result in results:
                if result in checked_ngrams:
                    continue
                term_ngrams = get_word_ngrams(result.split(" "), min_length=1, max_length=5)
                for ngram in term_ngrams:
                    checked_ngrams.add(ngram[0])
                filtered_results.append(result)
            results = filtered_results
            batch_facets.append(results)

        if not is_batch:
            return batch_facets[0]
        else:
            return batch_facets
