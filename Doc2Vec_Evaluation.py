
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
    
    Remaining_Persons = pickle.load(open(PATH_DATA + f'/Persons_Remaining.pkl', 'rb'))
    Remaining_Places = pickle.load(open(PATH_DATA + f'/Places_Remaining.pkl', 'rb'))
    
    # Compare different docs and words (via cosine_similarity) #                                                                                                                                     
    close_docs_words,_, cos_sim_matrix = return_the_closest_words_or_docs(word_doc="Grant",path_to_docs=path_to_docs,embeddings=final_embeddings,
                             new_dict=new_dict, dict_for_update=dict_of_tokens,
                             num_of_close_words=30, docs_only=False, token_only=False,
                             GPE_only=False, GPE_entities=Places_Remaining, PERSON_only=True,
                             PERSON_entities=Persons_Remaining, google_dict=False)
                             #GPE_only=False, GPE_entities=all_remaining_GPE_ents, PERSON_only=False,
                             #PERSON_entities=all_remaining_PERSON_ents, google_dict=False)
    
#    [return_the_closest_words_or_docs(word_doc=i,path_to_docs=path_to_docs,embeddings=final_embeddings,
#                             new_dict=new_dict, dict_for_update=dict_of_tokens,
#                             num_of_close_words=10, docs_only=True, token_only=False,
#                             GPE_only=False, GPE_entities=all_remaining_GPE_ents, PERSON_only=False,
#                             PERSON_entities=all_remaining_PERSON_ents, google_dict=False) for i in books_overview_names[:5]]
    
    
    
    google_embedding_matrix = pickle.load(open(PATH_DATA+ f'/google_embedding_matrix.pkl', 'rb'))

    def helper_for_doc_comparisons(doc_index, embeddings, google_dict):
        
        close_docs_words,_, cos_sim_matrix = return_the_closest_words_or_docs(word_doc=books_overview_names[doc_index],path_to_docs=path_to_docs,embeddings=embeddings,
                             new_dict=new_dict, dict_for_update=dict_of_tokens,
                             num_of_close_words=10, docs_only=True, token_only=False,
                             GPE_only=False, GPE_entities=all_remaining_GPE_ents, PERSON_only=False,
                             PERSON_entities=all_remaining_PERSON_ents, google_dict=google_dict)
        return dict(close_docs_words)
        
#    collection_of_doc_comparisons = [helper_for_doc_comparisons(i, embeddings=google_embedding_matrix, google_dict=True) for i in range(0, len(books_overview))]
#    collection_of_doc_comparisons_doc2vec = [helper_for_doc_comparisons(i, embeddings=final_embeddings, google_dict=False) for i in range(0, len(books_overview))]
#
#    
#    
#    collection_docs_sven = zip(books_overview_names, collection_of_doc_comparisons)
#    test1, test2 = zip(*collection_docs_sven)
#    pickle.dump(collection_docs_sven, open(PATH_DATA + f'/google_doc_comparisons.pkl', 'wb'))
    
    
    #check = cos_sim_matrix.loc[all_remaining_GPE_ents, all_remaining_GPE_ents]
    
    #pickle.dump(books_overview, open(PATH_DATA + f'/remaining_book_titles.pkl', 'wb'))
    #pickle.dump(cos_sim_matrix, open(PATH_DATA + f'/spacy_cos_sim_matrix.pkl', 'wb'))
    
    raw_token_collection = pickle.load(open(PATH_DATA+ f'/raw_tokens.pkl', 'rb'))
    """
    Erstelle eine Liste mit Tokens pro Dokument, die GPE oder eine Location sind --> danach z.B. Locations pro Buch oder Personen pro Buch bzw. ähnlichste 
    """
    path_to_entities = PATH_ENTITIES    #pickle.load(open(PATH_ENTITIES+ f'/raw_tokens.pkl', 'rb'))
    entities = natsort.natsorted(os.listdir(path_to_entities))
    extracted_entities = [pickle.load(open(PATH_ENTITIES+f'/'+str(i), 'rb')) for i in entities]
    
    """
    Global: TSNE(2D)(done!); Wort --> ähnlichste Dokumente; Häufigste Personen und Locations (global)  
    
    Lokal:
    Personen und GPE-Entities pro Dokument ist schon gefiltert --> nachfolgende Zeilen; Sven die 10 häufigsten Locations und Personen pro Buch; 
    den zugehörigen Buchtitel noch in den Dataframe als ID für Sven bereitstellen; Dokument --> ähnlichste Dokumente
    """
    GPE_entities_ = [[i[0] for i in j if i[1]=="GPE" and i[0] in dict_of_tokens.keys()] for j in extracted_entities]
    PERSON_entities_ = [[i[0] for i in j if i[1]=="PERSON" and i[0] in dict_of_tokens.keys()] for j in extracted_entities]

    All_GPE_entities_,_ = zip(*[i for j in extracted_entities for i in j if i[1]=="GPE"])
    All_PERSON_entities_,_ = zip(*[i for j in extracted_entities for i in j if i[1]=="PERSON"])
    ALL_GPEs_remaining = [i for i in All_GPE_entities_ if i in dict_of_tokens.keys()]
    ALL_PERSONS_remaining = [i for i in All_PERSON_entities_ if i in dict_of_tokens.keys()]
    
    """
    Kombinierte Tokenliste von allen Dokumenten
    """
#    list_of_preprocessed_token = raw_token_collection.copy()
#    list_with_all_tokens = [i for j in list_of_preprocessed_token for i in j] # creates one big list out of all docs
#    
     
    """
    hier noch falsch einsortierte GPEs und Persons über Häufigkeiten aussortieren; über NLTK FreqDist noch eine Liste mit den häufigsten Entities über alle Dokumente erstellen und danach diese Liste mit zugehörigen Häufigkeit pro  
    """
    GPE_most_frequent,_ = zip(*[i for i in nltk.FreqDist(All_GPE_entities_[:]).most_common(1000)])  # requires words from all docs 
    PERSON_most_frequent,_ = zip(*[i for i in nltk.FreqDist(ALL_PERSONS_remaining[:]).most_common(1000)]) # #1 is Thomas Lee (Confederate Army States General)

    all_remaining_GPE_ents = list(set([i for i in GPE_most_frequent if i in dict_of_tokens.keys()]))
    all_remaining_PERSON_ents = list(set([i for i in PERSON_most_frequent if i in dict_of_tokens.keys()]))
   
    #test = nltk.FreqDist(All_PERSON_entities_[:]).most_common(500)
    GPE_per_Book = [dict(nltk.FreqDist(i[:]).most_common(50)) for i in GPE_entities_]
    Person_per_Book = [dict(nltk.FreqDist(i[:]).most_common(10)) for i in PERSON_entities_]
    
    GPE_overall = (nltk.FreqDist(ALL_GPEs_remaining[:]).most_common(50))  #for i in ALL_GPEs_remaining]
    Person_overall = (nltk.FreqDist(ALL_PERSONS_remaining[:]).most_common(50)) #for i in ALL_PERSONS_remaining]
    
    """
    Nur Tags mit Vor- und Nachnamen (über alle Bücher finden) --> Regular Expression bauen die AbrahamLincoln wieder in AbrahamLincoln bzw. vor allem finden
    Sven schickt noch die Namen und Places aus den XML-Dateien um sie mit der Spacy-Liste abzugleichen 
    all_remaining_GPE_ents und all_remaining_PERSON_ents als Parameter in der Methode eventuell noch durch die aktualisierte Liste ersetzen
    """
    Svens_Places = pickle.load(open(PATH_DATA+"/overall_places.pkl", "rb"))
    Svens_Places_updated = [re.sub("\s+", "", i) for i in Svens_Places]
    Svens_Person = pickle.load(open(PATH_DATA+"/overall_persons.pkl", "rb"))
    Svens_Persons_updated = [re.sub("\s+", "", i) for i in Svens_Person]
    
    Full_Names_Remaining = [i for i in ALL_PERSONS_remaining if len(re.findall("[A-Z]+[a-z]+", str(i)))>1 and len(re.findall("Mc", str(i)))==0 and len(re.findall("Van", str(i)))==0]
    Persons_Remaining = [i for i in list(set(Full_Names_Remaining)) if i in Svens_Persons_updated]
    
    Places_Remaining = [i for i in list(set(ALL_GPEs_remaining)) if i in Svens_Places_updated]   
    
    #pickle.dump(Persons_Remaining, open(PATH_DATA + f'/Persons_Remaining.pkl', 'wb'))
    #pickle.dump(Places_Remaining, open(PATH_DATA + f'/Places_Remaining.pkl', 'wb'))

    

#    overall_ents = zip((GPE_overall), (Person_overall))
#    pickle.dump(overall_ents, open(PATH_DATA + f'/50Sven_overall_entities.pkl', 'wb'))
#    test3 = pickle.load(open(PATH_DATA + f'/50Sven_overall_entities.pkl', 'rb'))
#    test4, test5 = zip(*(test3))
#    test4 = dict(test4)
    
    #books_overview = pickle.load(open(PATH_DATA + f'/remaining_book_titles.pkl', 'rb'))
    
#    Sven_local_entities = zip(books_overview, GPE_per_Book, Person_per_Book)
#    pickle.dump(Sven_local_entities, open(PATH_DATA + f'/50Sven_local_entities.pkl', 'wb'))
    #Sven_books, Sven_GPE, Sven_Persons = zip(*Sven_local_entities)
    
#    files_in_directory_sorted = natsort.natsorted(os.listdir(path_to_docs))
#    #docs_sorted = [(Book_Titles[int(re.findall("\d+", i)[0])], str(index)) for index, i in enumerate(files_in_directory_sorted)]
#    docs_sorted = [int(re.findall("\d+", i)[0]) for index, i in enumerate(files_in_directory_sorted)]
#    pickle.dump(docs_sorted, open(PATH_DATA + f'/Book_IDs.pkl', 'wb'))
#    
#    
#    doc_comparisons = pickle.load(open(PATH_DATA + f'/doc_comparisons.pkl', 'rb'))
#    check = list(doc_comparisons)
#    check1, check2 = zip(*doc_comparisons)
    
    
    
    
    


    