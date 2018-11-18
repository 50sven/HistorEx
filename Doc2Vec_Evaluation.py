# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import re
import os
import natsort
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from CONSTANTS import *

"""
This method, if not yet done, adds the docs to the word-dictionary, computes the cosine-similarity matrix of the dictionary
and returns a specified number of nearest docs/words to the given doc/word according to the cosine-similarity
 
    Args:
        word_doc (string): a word or doc of which the "closest" words /docs should be computed
        path_to_docs (string): path to the docs (required to extract the doc-names)
        embeddings (matrix): contains all the vectors for all words and docs
        new_dict (dictionary or None): placeholder for the new dictionary to be created (which contains words and doc-names)
        dict_for_update (dictionary): dictionary containing all words
        num_of_close_words: the number of nearest words/docs that should be returned
        docs_only (boolean): True, if only the nearest docs should be returned
        

    Returns:
        A pandas-series containing the nearest words/docs and their corresponding cosine-similarity
"""

def return_the_closest_words_or_docs(word_doc, path_to_docs, embeddings, new_dict, dict_for_update, num_of_close_words, docs_only):
    
    if (new_dict is None): #  if this condition is true, the docs have to be added to the dictionary
        new_dict = [("doc_"+ str(i), i) for i in range(num_of_docs)] 
        files_in_directory_sorted = natsort.natsorted(os.listdir(path_to_docs))
        docs_sorted = [(re.sub(".txt", "", i), str(index)) for index, i in enumerate(files_in_directory_sorted)]
        new_dict = {key: value for (key, value) in docs_sorted} 
        new_dict.update(dict_for_update)
    
    vector_dataframe = pd.DataFrame(embeddings)
    cos_sim_matrix = pd.DataFrame(cosine_similarity(vector_dataframe, vector_dataframe))
    cos_sim_matrix.columns = new_dict.keys()
    cos_sim_matrix.index = new_dict.keys()
    
    if (docs_only):
        cos_sim_matrix_only_docs = cos_sim_matrix.iloc[:len(files_in_directory_sorted), :len(files_in_directory_sorted)]
        return cos_sim_matrix_only_docs[str(word_doc)].sort_values(ascending=False)[:num_of_close_words]
    
    return cos_sim_matrix[str(word_doc)].sort_values(ascending=False)[:num_of_close_words]


################################################################################################################
    

if __name__=="__main__":


    num_of_docs, context_window, dict_of_tokens, doc_context_indices_array, label_indices_array, freq_word_indices, unique_tokens = pickle.load(open(PATH_PICKLE_FILES + f'/final_input_list.pkl', 'rb'))
    final_embeddings = pickle.load(open(PATH_EMBEDDINGS+ f'/embeddings_epoch_8.pkl', 'rb')) # Training-Results after 8 Epochs
    path_to_docs = PATH_RAW_TEXT # imported from CONSTANTS.py
    new_dict = None
    
    
    # Compare different docs and words (via cosine_similarity)
    close_docs_words = return_the_closest_words_or_docs(word_doc="hero",path_to_docs=path_to_docs,embeddings=final_embeddings,
                             new_dict=new_dict, dict_for_update=dict_of_tokens,
                             num_of_close_words=10, docs_only=False)
    
    
    
    
    