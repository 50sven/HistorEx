# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from nltk.corpus import stopwords
from scipy.sparse import *
from scipy import *
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity



"""
Only the training of doc2vec --> Pre-Processing is done in a different file!
"""


#### Build the neural net ####

def train_neural_net(num_of_docs, context_window,dict_of_tokens,
                     doc_context_indices_array, label_indices_array, 
                     freq_word_indices, num_of_epochs, batch_size,
                     embedding_size, num_sampled):
    
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
                
                #writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
                saver = tf.train.Saver()
                with tf.Session(config=config) as sess:
                    #writer = tf.summary.FileWriter('./graphs', sess.graph)
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
                            if (i%2==0 and j==5):
                                
                                overall_loss = overall_loss/(iters*2)
                                if (i!=0):
                                    print ("Loss in epoch {} : {:5.2f}".format(i, overall_loss))
                                
                                overall_loss= 0
                                
                                if (i%2==0):
                                    sim = similarity.eval()
                        
                                    for m in range(0, len(freq_word_indices)): # the Wordlist is e.g. 16 Words long (gets specified as a parameter in the Preparation) --> the 16 words refer to a sample of the 100 most frequent words
                        
                                        valid_word = unique_word_list[freq_word_indices[m]-num_of_docs]
                                        top_k = 10  # number of nearest neighbors
                                        nearest = (-sim[m, :]).argsort()[1:top_k + 1] # 
                                        log_str = "Nearest to %s:" % valid_word
                        
                                        for k in range(top_k):
                        
                                            close_word = unique_word_list[nearest[k]-num_of_docs]
                                            log_str = "%s %s," % (log_str, close_word) # appends all current k-nearest neighbours to the frequent word
                        
                                        print(log_str)
                                pickle.dump(normalized_embeddings.eval(), open('/content/drive/My Drive/ISE_Training/weights_epoch_'+str(i)+'.pkl', 'wb'))
                                saver.save(sess,"./training_after_epoch_"+str(i)+".ckpt") 
                                
            
                    final_embeddings = normalized_embeddings.eval()
            
    return final_embeddings


################################################################################################################

if __name__=="__main__":
        
        
    ##### Training of the Neural Net #####
    
    num_of_docs, context_window, dict_of_tokens, doc_context_indices_array, label_indices_array, freq_word_indices, unique_tokens = pickle.load(open(f'/content/drive/My Drive/ISE_Training/final_input_list.pkl', 'rb'))
    
    
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
    
    

################################################################