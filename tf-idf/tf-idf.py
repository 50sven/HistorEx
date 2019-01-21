import pickle
from CONSTANTS import *
from nltk.corpus import stopwords
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import numpy as np
from gensim.models import TfidfModel
from collections import defaultdict
import re
from gensim import corpora, similarities
import pandas as pd

# copied from michi (doc2vec_preprocessing) and adapted to my purpose
def select_most_freq_words(list_of_prep_token, num_of_freq_words):
    list_of_preprocessed_token = list_of_prep_token.copy()
    list_with_all_tokens = [i for j in list_of_preprocessed_token for i in j]  # creates one big list out of all docs
    # length of all tokens = 33 749 279

    single_words_most_frequent = nltk.FreqDist(list_with_all_tokens[:]).most_common(
        num_of_freq_words - 1)  # requires words from all docs
    #
    non_tuple_most_frequent = [word[0] for word in
                               single_words_most_frequent]  # keep only the tokens without the corresponding frequencies

    # create one long list of tuples (token, index_of_the_corresponding_doc)
    #list_of_preprocessed_token = [
    #    (re.sub(i, "noise_word", i), doc_index) if i not in non_tuple_most_frequent else (i, doc_index) for doc_index, j
    #    in enumerate(list_of_preprocessed_token) for i in j]

    return non_tuple_most_frequent, len(list_of_prep_token), num_of_freq_words


# copied from michi
def further_pre_processing(processed_text_string, lemmatization):
    single_words = nltk.word_tokenize(processed_text_string)
    single_words = [word.lower() for word in single_words]

    if lemmatization == True:
        lemmatizer = WordNetLemmatizer()
        single_words = [lemmatizer.lemmatize(word) for word in single_words]

    cached_stopwords = stopwords.words('english')
    single_words_processed_fast = [word for word in single_words if word not in cached_stopwords]
    each_book = " ".join(single_words_processed_fast) # added this line to michis code to get list with raw books

    return each_book


if __name__ == '__main__':
    num_of_freq_words=10000

    raw_token_collection = pickle.load(open(PATH_DATA + f'/raw_tokens.pkl', 'rb')) # each doc is a list of raw tokens

    most_frequent_tokens, num_of_docs, num_of_tokens = select_most_freq_words(list_of_prep_token=raw_token_collection, num_of_freq_words= num_of_freq_words)
    books_with_most_freq_tokens = [[word for word in book if word not in most_frequent_tokens] for book in raw_token_collection]
    pickle.dump(books_with_most_freq_tokens, open(PATH_DATA + f'/books_with_most_freq_tokens.pkl', 'wb'))
    books_with_most_freq_tokens = pickle.load(open(PATH_DATA + f'/books_with_most_freq_tokens.pkl', 'rb'))

    # prepped_texts = pickle.load(open(PATH_DATA + f'/prepped_texts.pkl', 'rb'))

    # text_as_list as input for sklearn tf-idf
    # text_as_list = [further_pre_processing(processed_text_string=i, lemmatization=True) for i in prepped_texts]
    # pickle.dump(text_as_list, open(PATH_DATA + f'/text_as_list.pkl', 'wb'))
    # text_as_list = pickle.load(open(PATH_DATA + f'/text_as_list.pkl', 'rb'))


    ##### GENSIM PREPROCESSING
    # remove words that appear only once
    frequency = defaultdict(int)
    for text in raw_token_collection:
        for token in text:
            frequency[token]+=1
    text_freq = [[token for token in text if frequency[token]>1] for text in raw_token_collection]

    # dictionary with each token of whole corpus getting an id
    dictionary = corpora.Dictionary(text_freq)
    pickle.dump(dictionary, open(PATH_DATA + f'/dictionary.pkl', 'wb'))
    dictionary = pickle.load(open(PATH_DATA + f'/dictionary.pkl', 'rb'))

    dictionary_most_frequent = corpora.Dictionary(books_with_most_freq_tokens)
    pickle.dump(dictionary_most_frequent, open(PATH_DATA + f'/dictionary_most_frequent.pkl', 'wb'))
    dictionary_most_frequent = pickle.load(open(PATH_DATA + f'/dictionary_most_frequent.pkl', 'rb'))
    # #print(dictionary.token2id)
    #
    # # sparse matrix of books
    # books_in_sparse_matrix = [dictionary_most_frequent.doc2bow(text) for text in books_with_most_freq_tokens]
    # # pickle.dump(books_in_sparse_matrix, open(PATH_DATA + f'/books_in_sparse_matrix.pkl', 'wb')) # raw_token_collection -> text_freq -> dictionary
    books_in_sparse_matrix = pickle.load(open(PATH_DATA + f'/books_in_sparse_matrix.pkl', 'rb'))
    #
    ##### GENSIM TFIDF
    # SMART information Retrieval System (Smartirs): l - logarithm term frequency, t - idf, c - cosine normalization
    tfidf_gen = TfidfModel(corpus=books_in_sparse_matrix, smartirs='ltc')
    # number of features: 121 039 with dictionary (all tokens - stopwords - words that appear only once) ; raw_token_collection -> text_freq -> dictionary
    # number of features: 176 762 with dictionary_most_frequent (all tokens - stopwords - words that are frequent 10 000 words ) raw_token_collection -> books_with_most_freq_tokens -> dictionary_most_frequent
    index = similarities.SparseMatrixSimilarity(tfidf_gen[books_in_sparse_matrix], num_features=len(dictionary_most_frequent.token2id)) # transform whole corpus via tfidf and index it
    doc_similarites_most_frequent = np.array([index[tfidf_gen[book]] for book in books_in_sparse_matrix])

    #### get title of books
    # title = pd.read_csv(PATH_WIKIDATA + r'\1228_books_info_wiki.csv', sep='|')
    # title_tuple = [(nb, title) for nb, title in enumerate(title['Title'])]
    # pickle.dump(title_tuple, open(PATH_DATA + f'title_tuple.pkl', 'wb'))
    title_tuple = pickle.load(open(PATH_DATA + f'title_tuple.pkl', 'rb'))

    #### get max items to get max similarity and write to according format
    max_index = [list(book.argsort()[-11:][::-1]) for book in doc_similarites_most_frequent]
    # ten_sim_books_matrix = [ innerdic[valuej]=example_row.item(indexi, valuej) for indexi, valuei in enumerate(max_index) for indexj, valuej in enumerate(valuei) ]
    ten_sim_books_matrix = []
    for indexi, valuei in enumerate(max_index):
        innerdict = dict()
        for indexj, valuej in enumerate(valuei):
            #if int(indexi) != valuej:
            innerdict[title_tuple[valuej][1]] = doc_similarites_most_frequent.item(indexi, valuej)
        ten_sim_books_matrix.append(innerdict)
    pickle.dump(ten_sim_books_matrix, open(PATH_DATA + f'ten_sim_books_matrix.pkl', 'wb'))
    ten_sim_books_matrix = pickle.load(open(PATH_DATA + f'ten_sim_books_matrix.pkl', 'rb'))



    ##################################################################

    ###### SKLEARN
    # CountVectorizer --> how does outcome look like?
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(text_as_list)

    # TfidfTransormer
    transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    tfidf_skl = transformer.fit_transform(X)
    tfidf_array = tfidf_skl.toarray()





