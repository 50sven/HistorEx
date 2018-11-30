import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr


"""
Evaluation of the trained wordvectors vs. State-of-the-Art Databases (SimLex-999) for human evaluations of word similarities
More Details on SimLex-999 and Scores of State-of-the-Art Models: https://aclweb.org/aclwiki/SimLex-999_(State_of_the_art)

We compute Spearmans Rang-Correlation for the benchmark scores in the SimLex-999 Database and our trained wordvectors 

"""


num_of_docs, context_window, dict_of_tokens, doc_context_indices_array, label_indices_array, freq_word_indices, unique_tokens = pickle.load(open(PATH_DATA+ f'/final_input_list_spacy.pkl', 'rb'))


word_sim_database = pd.read_csv(PATH_DATA+"/SimLex-999.csv", sep="\t")

word_list =  list(word_sim_database["word1"]) +  list(word_sim_database["word2"])
word_list_word_1 = list(word_sim_database["word1"])

check = [1 if (i in dict_of_tokens.keys() and word_list[index+999] in dict_of_tokens.keys())  else 0 for index,i in enumerate(word_list_word_1)]
check_array = np.sum(np.asarray(check), axis=0) 

remaining_words = [ list(word_sim_database.iloc[index,:2]) for index,word in enumerate(check) if check[index]==1 ]
remaining_similarities = np.reshape(np.asarray([list(word_sim_database.iloc[index,3:4]) for index,word in enumerate(check) if check[index]==1 ]), [-1])


words1 = [i[0] for i in remaining_words]
words2 = [i[1] for i in remaining_words]


cos_sim_matrix = pickle.load(open(PATH_DATA + f'/spacy_cos_sim_matrix.pkl', 'rb'))


reference_similarities = cos_sim_matrix.loc[words1, words2]
collected_similarities = pd.Series(np.diag(reference_similarities), index=[reference_similarities.index, reference_similarities.columns])


spearmanr(np.asarray(collected_similarities.iloc[:]), remaining_similarities) # Spearmans Rho = 0.315 











