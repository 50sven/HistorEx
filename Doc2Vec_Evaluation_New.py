import numpy as np
import pandas as pd
import re
import os
import natsort
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from CONSTANTS import *
from matplotlib import pyplot as plt
import nltk


"""
This method, if not yet done, adds the docs to the word-dictionary, computes the cosine-similarity matrix of the dictionary
and returns a specified number of nearest docs/words to the given doc/word according to the cosine-similarity
Depending on the specification of the user, only the closest words kann be returned or only the closest documents
Additionally the user can return to closest documents, if any token is given (which can be any word as well), which enables a semantic search
 
    Args:
        word_doc (string): a word or doc of which the "closest" words /docs should be computed
        path_to_docs (string): path to the docs (required to extract the doc-names)
        embeddings (matrix): contains all the vectors for all words and docs
        new_dict (dictionary or None): placeholder for the new dictionary to be created (which contains words and doc-names)
        dict_for_update (dictionary): dictionary containing all words
        num_of_close_words: the number of nearest words/docs that should be returned
        docs_only (boolean): True, if only the nearest docs should be returned
        token_only (boolean): True, if only the nearest words should be returned
        

    Returns:
        1. Argument A pandas-series containing the nearest words/docs and their corresponding cosine-similarity
        2. Argument: Returns a list of the used books 
        3. Argument: Returns a DataFrame which contains all pairwise cosine-similarities
"""



def return_the_closest_words_or_docs(word_doc, path_to_docs, embeddings, new_dict, dict_for_update,
                                     num_of_close_words, docs_only, token_only, GPE_only, GPE_entities,
                                     PERSON_only, PERSON_entities, google_dict):
    
    if (new_dict is None): #  if this condition is true, the docs have to be added to the dictionary
        
        files_in_directory_sorted = natsort.natsorted(os.listdir(path_to_docs))
        docs_sorted = [(Book_Titles[int(re.findall("\d+", i)[0])], str(index)) for index, i in enumerate(files_in_directory_sorted)]
        titles, _ = zip(*docs_sorted)
        docs_sorted = [(str(titles[index])+"_PART_2",index) if (index>0 and titles[index] == titles[index-1]) else (titles[index],index) for index,i in enumerate(docs_sorted)] # if titles[index]==titles[index-1]] # 
        
        doc_names,_ = zip(*docs_sorted)
        
        
        new_dict = {key: value for (key, value) in docs_sorted}
        if (google_dict==False):
            new_dict.update(dict_for_update)
        
    vector_dataframe = pd.DataFrame(embeddings)
    cos_sim_matrix = pd.DataFrame(cosine_similarity(vector_dataframe, vector_dataframe))
    cos_sim_matrix.columns = new_dict.keys()
    cos_sim_matrix.index = new_dict.keys()
    
    
    if (GPE_only):
        cos_sim_matrix_only_GPE = cos_sim_matrix.loc[list(set(GPE_entities+[str(word_doc)])), list(set(GPE_entities+[str(word_doc)]))]
        return cos_sim_matrix_only_GPE[str(word_doc)].sort_values(ascending=False)[:num_of_close_words], dict(docs_sorted), cos_sim_matrix
    
    elif (PERSON_only):
        cos_sim_matrix_only_PERSON = cos_sim_matrix.loc[list(set(PERSON_entities+[str(word_doc)])), list(set(PERSON_entities+[str(word_doc)]))]
        return cos_sim_matrix_only_PERSON[str(word_doc)].sort_values(ascending=False)[:num_of_close_words], dict(docs_sorted), cos_sim_matrix

        
    elif (docs_only==True and token_only==False):
        
        if (str(word_doc) in list(doc_names)):
            cos_sim_matrix_only_docs = cos_sim_matrix.iloc[:len(files_in_directory_sorted), :len(files_in_directory_sorted)] 
        else:
            word_index = dict_of_tokens[str(word_doc)]
            cos_sim_matrix_only_docs = cos_sim_matrix.iloc[(list(range(0,len(files_in_directory_sorted)))+[word_index]),(list(range(0,len(files_in_directory_sorted)))+[word_index])]
        
        return cos_sim_matrix_only_docs[str(word_doc)].sort_values(ascending=False)[:num_of_close_words], dict(docs_sorted), cos_sim_matrix
    
    elif (token_only==True and docs_only==False):
        
        if (str(word_doc) in list(doc_names)):
            word_index = new_dict[str(word_doc)]
            print(word_index)
            cos_sim_matrix_only_tokens = cos_sim_matrix.iloc[(list(range(len(files_in_directory_sorted),len(cos_sim_matrix.columns)))+[word_index]),(list(range(len(files_in_directory_sorted),len(cos_sim_matrix.columns)))+[word_index])]
            #return cos_sim_matrix_only_tokens[str(word_doc)].sort_values(ascending=False)[:num_of_close_words], dict(docs_sorted), cos_sim_matrix
        else: 
            cos_sim_matrix_only_tokens = cos_sim_matrix.iloc[len(files_in_directory_sorted):, len(files_in_directory_sorted):]
        return cos_sim_matrix_only_tokens[str(word_doc)].sort_values(ascending=False)[:num_of_close_words], dict(docs_sorted), cos_sim_matrix

    
    else: return cos_sim_matrix[str(word_doc)].sort_values(ascending=False)[:num_of_close_words], dict(docs_sorted), cos_sim_matrix


################################################################################################################
    

if __name__=="__main__":


    num_of_docs, context_window, dict_of_tokens, doc_context_indices_array, label_indices_array, freq_word_indices, unique_tokens = pickle.load(open(PATH_DATA+ f'/final_input_list_spacy.pkl', 'rb'))
     
    final_embeddings = pickle.load(open(PATH_EMBEDDINGS+ f'/Spacy_300dim_embeddings_epoch_8.pkl', 'rb')) # Training-Results after 8 Epochs
    path_to_docs = PATH_RAW_TEXT # imported from CONSTANTS.py
    new_dict = None
    
    Book_Titles = pickle.load(open(PATH_BOOK_TITLES+ f'/Book_Titles.pkl', 'rb')) # get the actual book titles
    books_overview = pickle.load(open(PATH_DATA + f'/remaining_book_titles.pkl', 'rb'))
    books_overview_names = list(books_overview.keys())
    
    Persons_Remaining = pickle.load(open(PATH_DATA + f'/Persons_Remaining.pkl', 'rb'))
    Places_Remaining = pickle.load(open(PATH_DATA + f'/Places_Remaining.pkl', 'rb'))
    
    # Compare different docs and words (via cosine_similarity) #                                                                                                                                     
    close_docs_words,_, cos_sim_matrix = return_the_closest_words_or_docs(word_doc="Grant",path_to_docs=path_to_docs,embeddings=final_embeddings,
                             new_dict=new_dict, dict_for_update=dict_of_tokens,
                             num_of_close_words=30, docs_only=False, token_only=False,
                             GPE_only=False, GPE_entities=Places_Remaining, PERSON_only=True,
                             PERSON_entities=Persons_Remaining, google_dict=False)
    
    Full_Names = [" ".join(re.findall("[A-Z]+[a-z]+", i)) for i in close_docs_words.index] # Sur- and Familynames separated by a space
    
