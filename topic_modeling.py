import lda
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import pyLDAvis.gensim as gensimvis
import pyLDAvis
nltk.download('wordnet')


def lemmatize_stem(text):
    """
    Lemmatizes & stems text.
    """

    stemmer = SnowballStemmer("english")
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    """
    Tokenizes text using gensim, removes stop words & words with fewer than 3 characters, then 
    calls lemmatize_stem on the token.
    """

    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stem(token))
    return result


def lda_model(corpus, dictionary):
    """
    LDA Topic Modeling, saves model to an interactive html file.
    """

    lda_model = gensim.models.LdaMulticore(
        corpus, num_topics=4, id2word=dictionary, passes=2, workers=3)

    # Visualize LDA output using pyLDAvis plot
    movies_vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    movie_lda = open('movie_lda.html', 'w')
    pyLDAvis.save_html(movies_vis_data, movie_lda)

    for idx, topic in lda_model.print_topics(num_topics=-1, num_words=15):
        print('Topic: {} \nWords: {}'.format(idx, topic))


def main():
    data = pd.read_csv('Collecting_Data/all_movies.csv')

    # Extract the column 'overview' into a new dataframe
    text_data = data[['overview']]
    text_data['index'] = text_data.index

    # Preprocess the 'overview' text, saving the results as 'processed_docs'
    processed_docs = text_data['overview'].map(preprocess)

    # Add the processed results to a gensim dictionary
    dictionary = gensim.corpora.Dictionary(processed_docs)

    # Filter out tokens that appear in less than 3 documents or more than 90% of the entries
    dictionary.filter_extremes(no_below=3, no_above=0.9)

    # Convert document into the bag-of-words (BoW) format = list of (token_id, token_count) tuples.
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    lda_model(bow_corpus, dictionary)


if __name__ == "__main__":
    main()
