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
import spacy



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


def convert_ngram_to_unigram (book_i):
    
    n = 100000
    
    splitted_book = [nlp(book_i[i:i+n]) for i in range(0, len(book_i), n)] # Split to texts to enable nlp()-methode of spacy
    ents_list = [ list(i.ents) for i in splitted_book] 
    all_ents = sum(ents_list,[]) # all entities of one document 
    
    bigrams_ent_indices = [(re.sub(r"[^\w]","", str(i)), index) for index,i in enumerate(all_ents) if len(i)>1 and (i.label_=="GPE" or i.label_=="PERSON")] # Create the list, which contains selected entities as a unigram
    
    old_terms = []
    new_terms = []
    
    book_i = re.sub(r'[^\w]', ' ',book_i)
    for i,index in bigrams_ent_indices:
        
        old_term = re.sub(r'[^\w]', ' ',str(all_ents[index]))
        new_term = re.sub(r'[^\w]', ' ',str(i))
        
        book_i = re.sub(str(old_term), str(new_term), book_i) # replace the n-grams which represent selected entities with one token
        old_terms.append(all_ents[index])
        new_terms.append(i)
        
    book_i = re.sub("\s+", " ", book_i)
    print("Book_converted")

    return book_i   
 



####
def remove_unnecessary_chars (text_string):

    raw_text_string = text_string

    raw_text_string = raw_text_string.replace('\\n', " ")
    raw_text_string = raw_text_string.replace('\\r', " ")
    raw_text_string = re.sub("\s+", " ", raw_text_string)
    
    raw_text_string = convert_ngram_to_unigram(raw_text_string)  
    
    raw_text_string = re.sub("[^a-zA-Z]+", " ", raw_text_string)
    raw_text_string = re.sub(r"\b\w\b", " ", raw_text_string)
    raw_text_string = re.sub("\s+", " ", raw_text_string)

    return raw_text_string



#### Apply further pre_processing to the raw_text_string ####

def further_pre_processing(processed_text_string, lemmatization, to_lower):

    single_words = nltk.word_tokenize(processed_text_string)
    if to_lower==False:
        single_words = [word.lower() for word in single_words]
    
    if lemmatization==False:
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
    
    single_words_most_frequent = nltk.FreqDist(list_with_all_tokens[:]).most_common(num_of_freq_words)  # requires words from all docs 
    non_tuple_most_frequent = [word[0] for word in single_words_most_frequent] # keep only the tokens without the corresponding frequencies
    
    # create one long list of tuples (token, index_of_the_corresponding_doc)
    #list_of_preprocessed_token = [(re.sub(i, "noise_word", i), doc_index) if i not in non_tuple_most_frequent else (i,doc_index) for doc_index,j in enumerate(list_of_preprocessed_token) for i in j]
    list_of_preprocessed_token2 = [(i,doc_index) for doc_index,j in enumerate(list_of_preprocessed_token) for i in j  if i in non_tuple_most_frequent]

    
    return  single_words_most_frequent, len(list_of_prep_token), num_of_freq_words, list_of_preprocessed_token2 



# Create the input-tuples and further relevant information for the neural net #
    
def pre_training_preparation(token_doc_tuple, num_of_docs, half_context_size, num_of_samples, num_of_freq_words):

    token_doc_tuple = [i for i in token_doc_tuple if type(i) ==tuple] # Only use tuples (non-tuple implies an error)
    only_tokens , doc_ids = zip(*token_doc_tuple) # separate in doc-id's and tokens 
    
    unique_tokens = set(only_tokens) # Set of unique words across all documents 
    unique_tokens = list(zip(unique_tokens, range(num_of_docs, (num_of_docs+num_of_freq_words)))) # num_of_freq_words different words + the word "noise" -> indexing (start counting at num_of_docs (the numbers below num_of_docs are the doc indices))
    
    dict_of_tokens = {key: value for (key, value) in unique_tokens} # dictionary for the most frequent words + "noise" (the token for all less frequent words)
    words_to_indices = [dict_of_tokens[i] for i in only_tokens] # list comprehension, to receive the indices for all words from the dictionary
    
    
    #input_test_list = [[doc_ids[i]]+words_to_indices[i:(i+(half_context_size))]+words_to_indices[(i+(half_context_size+1)):(i+((2*half_context_size)+1))] for i in range(len(words_to_indices)-2*half_context_size)] # list with indices of the context words per input
    doc_context_indices_array = np.asarray([[doc_ids[i]]+words_to_indices[i:(i+(half_context_size))]+words_to_indices[(i+(half_context_size+1)):(i+((2*half_context_size)+1))] for i in range(len(words_to_indices)-2*half_context_size)]) # list with indices of the context words per input
    #doc_context_indices_array = np.asarray(input_test_list) # Matrix with indices, which serves as Input for the neural net
    
    label_indices_array = np.asarray([words_to_indices[half_context_size+i] for i in range(len(words_to_indices)-2*half_context_size)]) 
    #label_indices_array = np.asarray(label_list)
    
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
    num_of_freq_words=12500 
    #
        
    path_to_docs =  PATH_RAW_TEXT 
    texts, sorted_list = convert_docs_to_string(path_to_docs=path_to_docs, open_mode="rb", combine_docs=False)
    
    
    nlp = spacy.load('en_core_web_sm')
    prepped_texts = [remove_unnecessary_chars(i) for i in texts]
    
    raw_token_collection = [further_pre_processing(processed_text_string=i, lemmatization=False, to_lower=False) for i in prepped_texts]
    
    
    """
    The pickle-lines enable a sequential pre-processing --> different Preprocessing-Steps take a while and require a lot of resources
    """
    
    raw_token_collection = pickle.load(open(PATH_DATA+ f'/raw_tokens.pkl', 'rb'))
    
    most_frequent_words, num_of_docs, num_of_freq_words, token_doc_tuple = select_most_freq_words(raw_token_collection, num_of_freq_words=num_of_freq_words)
    most_frequent_words, num_of_docs, num_of_freq_words, token_doc_tuple = pickle.load(open(PATH_DATA+ f'/most_freq_words.pkl', 'rb'))
    
    
    num_of_docs, context_window, dict_of_tokens, doc_context_indices_array, label_indices_array, freq_word_indices, unique_tokens = pre_training_preparation(token_doc_tuple=token_doc_tuple, num_of_docs=num_of_docs,
                                                                                                                                              half_context_size=half_context_size, num_of_samples=num_of_samples,
                                                                                                                                              num_of_freq_words=num_of_freq_words)

    final_input_list_changed_spacy = (num_of_docs, context_window, dict_of_tokens, doc_context_indices_array, label_indices_array, freq_word_indices, unique_tokens)
    pickle.dump(final_input_list_changed_spacy, open(PATH_DATA + f'/final_input_list_spacy.pkl', 'wb'))
    
################################################################