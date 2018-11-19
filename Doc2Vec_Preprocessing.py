# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
import pickle
import natsort
import random
from CONSTANTS import *



"""
Different Pre-Processing Steps to prepare the Data for doc2vec-Training
"""

#### Pre-Processing of the documents -> Create inputs, which can be used in the doc2vec-Net ####

# Read-in the text #

def convert_docs_to_string(path_to_docs, open_mode, combine_docs):
    
    text_data = []
    directory_name = path_to_docs
    files_in_directory_sorted = natsort.natsorted(os.listdir(path_to_docs))
    
    
    for value in files_in_directory_sorted:
        file_name= directory_name+"/"+value
        data = open(file_name, open_mode) #, encoding=encoding_type)
        data_string = data.read()
        text_data.append(data_string) # pass whole String to List
    
    text_data= [str(i) for i in text_data] # avoid any byte-code
    if (combine_docs ==True):    
        whole_string_big = " ".join(text_data)
    else :
        return text_data, files_in_directory_sorted
    
    return  whole_string_big


####

def remove_unnecessary_chars (text_string):

    raw_text_string = text_string

    raw_text_string = raw_text_string.replace('\\n', " ")
    raw_text_string = raw_text_string.replace('\\r', " ")
    
    raw_text_string = re.sub("[^a-zA-Z]+", " ", raw_text_string)
    raw_text_string = re.sub(r"\b\w\b", " ", raw_text_string)
    raw_text_string = re.sub("\s+", " ", raw_text_string)

    return raw_text_string



#### Apply further pre_processing to the raw_text_string ####

def further_pre_processing(processed_text_string, lemmatization):

    single_words = nltk.word_tokenize(processed_text_string)
    single_words = [word.lower() for word in single_words]
    
    if lemmatization==True:
        lemmatizer = WordNetLemmatizer()
        single_words = [lemmatizer.lemmatize(word) for word in single_words]
    
    cached_stopwords = stopwords.words('english')
    single_words_processed_fast = [word for word in single_words if word not in cached_stopwords]

    return single_words_processed_fast

####

# Select the most frequent words for training & mark the less frequent words as Noise #
    
def select_most_freq_words (list_of_prep_token, num_of_freq_words):
    
    list_of_preprocessed_token = list_of_prep_token.copy()
    list_with_all_tokens = [i for j in list_of_preprocessed_token for i in j] # creates one big list out of all docs
    
    single_words_most_frequent = nltk.FreqDist(list_with_all_tokens[:]).most_common(num_of_freq_words-1) # requires words from all docs 
    non_tuple_most_frequent = [word[0] for word in single_words_most_frequent] # keep only the tokens without the corresponding frequencies
    
    # create one long list of tuples (token, index_of_the_corresponding_doc)
    list_of_preprocessed_token = [(re.sub(i, "noise_word", i), doc_index) if i not in non_tuple_most_frequent else (i,doc_index) for doc_index,j in enumerate(list_of_preprocessed_token) for i in j]
    
    
    return list_of_preprocessed_token, len(list_of_prep_token), num_of_freq_words



# Create the input-tuples and further relevant information for the neural net #
    
def pre_training_preparation(token_doc_tuple, num_of_docs, half_context_size, num_of_samples, num_of_freq_words):

    token_doc_tuple = [i for i in token_doc_tuple if type(i) ==tuple] # Only use tuples (non-tuple implies an error)
    only_tokens , doc_ids = zip(*token_doc_tuple) # separate in doc-id's and tokens 
    
    unique_tokens = set(only_tokens) # Set of unique words across all documents 
    unique_tokens = list(zip(unique_tokens, range(num_of_docs, (num_of_docs+num_of_freq_words)))) # num_of_freq_words different words + the word "noise" -> indexing (start counting at num_of_docs (the numbers below num_of_docs are the doc indices))
    
    dict_of_tokens = {key: value for (key, value) in unique_tokens} # dictionary for the most frequent words + "noise" (the token for all less frequent words)
    words_to_indices = [dict_of_tokens[i] for i in only_tokens] # list comprehension, to receive the indices for all words from the dictionary
    
    
    input_test_list = [[doc_ids[i]]+words_to_indices[i:(i+(half_context_size))]+words_to_indices[(i+(half_context_size+1)):(i+((2*half_context_size)+1))] for i in range(len(words_to_indices)-2*half_context_size)] # list with indices of the context words per input
    doc_context_indices_array = np.asarray(input_test_list) # Matrix with indices, which serves as Input for the neural net
    
    label_list = [words_to_indices[half_context_size+i] for i in range(len(words_to_indices)-2*half_context_size)] 
    label_indices_array = np.asarray(label_list)
    
    frequency_of_tokens = nltk.FreqDist(only_tokens[:])
    most_frequent_tokens,_ = zip(*(sorted(frequency_of_tokens.items(), key=lambda item: item[1])[-100:])) # the most frequent word is the last in most_frequent_tokens
    
    sample_of_freq_words = [most_frequent_tokens[np.random.choice(100,num_of_samples, replace=False)[i]] for i in range(num_of_samples)] 
    freq_word_indices = [dict_of_tokens[i] for i in sample_of_freq_words]
    
    inputs_labels = list(zip(doc_context_indices_array, label_indices_array))
    random.shuffle(inputs_labels)
    doc_context_indices_array, label_indices_array = zip(*inputs_labels)
    
    return num_of_docs, ((2*half_context_size)+1), dict_of_tokens, np.asarray(doc_context_indices_array), np.asarray(label_indices_array), freq_word_indices, unique_tokens


####
    
if __name__=="__main__":
        
        
    ##### Pre-Processing  #####
    
    # Parameters #
    half_context_size=3
    num_of_samples=16
    num_of_freq_words=10000 
    #
        
    path_to_docs =  PATH_RAW_TEXT 
    texts, sorted_list = convert_docs_to_string(path_to_docs=path_to_docs, open_mode="rb", combine_docs=False)
    
    prepped_texts = [remove_unnecessary_chars(i) for i in texts]
    raw_token_collection = [further_pre_processing(processed_text_string=i, lemmatization=True) for i in prepped_texts]
    
    """
    The pickle-lines enable a sequential pre-processing --> different Preprocessing-Steps take a while and require a lot of resources
    """
    
    #pickle.dump(raw_token_collection, open(PATH_DATA + f'/raw_tokens.pkl', 'wb'))
    #raw_token_collection = pickle.load(open(PATH_DATA+ f'/raw_tokens.pkl', 'rb'))
    
    
    token_doc_tuple, num_of_docs, num_of_freq_words = select_most_freq_words(raw_token_collection, num_of_freq_words=num_of_freq_words)
    
    
    #prep_list = (token_doc_tuple, num_of_docs, num_of_freq_words)
    #pickle.dump(prep_list, open(PATH_DATA +f'/prep_list.pkl', 'wb'))
    token_doc_tuple, num_of_docs, num_of_freq_words = pickle.load(open(PATH_DATA+ f'/prep_list.pkl', 'rb'))
    
    
    num_of_docs, context_window, dict_of_tokens, doc_context_indices_array, label_indices_array, freq_word_indices, unique_tokens = pre_training_preparation(token_doc_tuple=token_doc_tuple, num_of_docs=num_of_docs,
                                                                                                                                              half_context_size=half_context_size, num_of_samples=num_of_samples,
                                                                                                                                              num_of_freq_words=num_of_freq_words)

    #final_input_list = (num_of_docs, context_window, dict_of_tokens, doc_context_indices_array, label_indices_array, freq_word_indices, unique_tokens)
    #pickle.dump(final_input_list, open(PATH_DATA + f'/final_input_list.pkl', 'wb'))
    
################################################################