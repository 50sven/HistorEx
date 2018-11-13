# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 17:23:30 2018

@author: Michael
"""


import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
import re
import os
import random
from nltk.corpus import stopwords
from scipy.sparse import *
from scipy import *
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
#from CONSTANTS import *

"""
Manual Implementation of the doc2vec-Model 
"""

#### Pre-Processing of the documents -> Create inputs, which can be used in the doc2vec-Net ####

# Read-in the text #

def convert_docs_to_string(path_to_docs, open_mode, combine_docs):
    
    text_data = []
    directory_name = path_to_docs
    files_in_directory = os.listdir(path_to_docs)
    
    for value in files_in_directory:
        file_name= directory_name+"/"+value
        data = open(file_name, open_mode) #, encoding=encoding_type)
        data_string = data.read()
        text_data.append(data_string) # pass whole String to List
    
    text_data= [str(i) for i in text_data] # avoid any byte-code
    if (combine_docs ==True):    
        whole_string_big = " ".join(text_data)
    else :
        return text_data
    
    return whole_string_big


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
    list_of_preprocessed_token = [(re.sub(i, "noise", i), doc_index) if i not in non_tuple_most_frequent else (i,doc_index) for doc_index,j in enumerate(list_of_preprocessed_token) for i in j]
    
    
    return list_of_preprocessed_token, len(list_of_prep_token), num_of_freq_words




def pre_training_preparation(token_doc_tuple, num_of_docs, half_context_size, num_of_samples, num_of_freq_words):


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
    
    return num_of_docs, ((2*half_context_size)+1), dict_of_tokens, np.asarray(doc_context_indices_array), np.asarray(label_indices_array), freq_word_indices


####
  
"""
die Inputs am besten vor dem Training oder einmal pro Epoche shuffeln
"""    


#### Build the neural net ####

def train_neural_net(num_of_docs, context_window,dict_of_tokens,
                     doc_context_indices_array,
                     label_indices_array, freq_word_indices, num_of_epochs, batch_size, embedding_size, num_sampled):
    
    overall_loss= 0
    label_indices_array = np.reshape(label_indices_array, [-1,1])
    num_classes = len(dict_of_tokens)
    unique_word_list = list(dict_of_tokens)
    graph1 = tf.get_default_graph()
    
    with graph1.as_default():
        
        
        with tf.device('/gpu:0'):   
            
                train_inputs = tf.placeholder(tf.int32, shape=[None, context_window])
                train_labels = tf.placeholder(tf.int32, shape=[None,1]) 
                freq_word_indices_tensor = tf.constant(freq_word_indices, dtype=tf.int32) # Indexpositions of the sampled frequent words
                        
                
                word_embeddings = tf.Variable(tf.random_uniform([len(dict_of_tokens)+num_of_docs, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(word_embeddings,train_inputs) # the Lookup-Method selects rows of the doc/word-embedding matrices and uses train_inputs as the necessary indices
                mean_context_embed = tf.reduce_mean(embed,1) # create one row as the average of the selected context-embeddings(context-rows) and the doc-embedding(doc_i row)
                
                nce_weights = tf.Variable(tf.truncated_normal([len(dict_of_tokens), embedding_size], # Weight-Matrix between Hidden- and Outputlayer
                                                               stddev= 1.0/ math.sqrt(embedding_size)))
                nce_biases = tf.Variable(tf.zeros([len(dict_of_tokens)])) # Bias-Vector for the Output-Layer
                
                # Compute the cosinus similarity (between minibatch examples and all embeddings.) #
            
                norm = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keepdims=True))
                normalized_embeddings = word_embeddings / norm
                valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, freq_word_indices_tensor) 
                similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
                
                            
                nce_loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels,
                                                         mean_context_embed, num_sampled, num_classes))
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                train = optimizer.minimize(nce_loss)
                        
                init = tf.global_variables_initializer()
                
                config = tf.ConfigProto(allow_soft_placement = True) 
                config.gpu_options.per_process_gpu_memory_fraction = 0.8
                
                writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
                with tf.Session(config=config) as sess:
                    writer = tf.summary.FileWriter('./graphs', sess.graph)
                    sess.run(init)
                    
                    iters = int(np.floor((label_indices_array.shape[0])/batch_size)) # number of batches in the Training Data
                    
                    for i in range(num_of_epochs):
                        for j in range(iters):
                            
                            current_batch = doc_context_indices_array[(j*batch_size):((j+1)*batch_size),:] # select the next batch of the Training Data
                            current_labels = label_indices_array[(j*batch_size):((j+1)*batch_size), :] # return the labels of the current batch
                            
                            _,loss_val = sess.run([train, nce_loss], feed_dict={train_inputs:current_batch, train_labels: current_labels})
                            overall_loss = loss_val + overall_loss
                            
                            if ((i==0) and (j==0)):
                                print("Training has started...")
                                print("\n")
                            if (i%10==0 and j==5):
                                
                                overall_loss = overall_loss/(iters*10)
                                if (i!=0):
                                    print ("Loss in epoch {} : {:5.2f}".format(i, overall_loss))
                                
                                overall_loss= 0
                                
                                if (i%20==0):
                                    sim = similarity.eval()
                        
                                    for m in range(0, len(freq_word_indices)): # the Wordlist is e.g. 16 Words long (gets specified as a parameter in the Preparation) --> the 16 words refer to a sample of the 100 most frequent words
                        
                                        valid_word = unique_word_list[freq_word_indices[m]-num_of_docs]
                        
                                        top_k = 8  # number of nearest neighbors
                        
                                        nearest = (-sim[m, :]).argsort()[1:top_k + 1] # 
                        
                                        log_str = "Nearest to %s:" % valid_word
                        
                                        for k in range(top_k):
                        
                                            close_word = unique_word_list[nearest[k]-num_of_docs]
                        
                                            log_str = "%s %s," % (log_str, close_word) # appends all current k-nearest neighbours to the frequent word
                        
                                        print(log_str)
            
                    final_embeddings = normalized_embeddings.eval()
            
    return final_embeddings


def return_the_closest_words(word, embeddings, dictionary, num_of_close_words):
    
    new_dict = [("doc_"+ str(i), i) for i in range(num_of_docs)]
    new_dict = {key: value for (key, value) in new_dict} 
    new_dict.update(dictionary)
    
    vector_dataframe = pd.DataFrame(embeddings)
    cos_sim_matrix = pd.DataFrame(cosine_similarity(vector_dataframe, vector_dataframe))
    cos_sim_matrix.columns = new_dict.keys()
    cos_sim_matrix.index = new_dict.keys()
    
    return cos_sim_matrix[str(word)].sort_values(ascending=False)[:num_of_close_words]


################################################################################################################

if __name__=="__main__":
        
        
    ##### Pre-Processing + Training of the Neural Net #####
    
    # Parameters #
    half_context_size=4
    num_of_samples=16
    num_of_freq_words=3000
    #
        
    path_to_docs = "/content/ISE/all_books/"          #PATH_CONTENT
    texts = convert_docs_to_string(path_to_docs=path_to_docs, open_mode="rb", combine_docs=False)
    
    prepped_texts = [remove_unnecessary_chars(i) for i in texts]
    raw_token_collection = [further_pre_processing(processed_text_string=i, lemmatization=True) for i in prepped_texts]
    
    
    token_doc_tuple, num_of_docs, num_of_freq_words = select_most_freq_words(raw_token_collection, num_of_freq_words=num_of_freq_words)
    num_of_docs, context_window, dict_of_tokens, doc_context_indices_array, label_indices_array, freq_word_indices = pre_training_preparation(token_doc_tuple=token_doc_tuple, num_of_docs=num_of_docs,
                                                                                                                                              half_context_size=half_context_size, num_of_samples=num_of_samples,
                                                                                                                                              num_of_freq_words=num_of_freq_words)

    
    # Hyperparameters #
    embedding_size = 100                    
    num_sampled= 64
    num_of_epochs = 100
    batch_size = 512    
    #   
    
    final_embeddings = train_neural_net(num_of_docs=num_of_docs, context_window=context_window,dict_of_tokens=dict_of_tokens,
                                        doc_context_indices_array=doc_context_indices_array,
                                        label_indices_array=label_indices_array, freq_word_indices=freq_word_indices,
                                        num_of_epochs=num_of_epochs, batch_size=batch_size, embedding_size=embedding_size, num_sampled= num_sampled)
    
    
    # Compare different docs and words (via cosine_similarity)
    
    #return_the_closest_words(word="hour", embeddings=final_embeddings, dictionary=dict_of_tokens, num_of_close_words=10)

################################################################