"""
Visualization of Document-Vectors with the TSNE-Method
"""

import numpy as np
from sklearn.manifold import TSNE
from CONSTANTS import *
from matplotlib import pyplot as plt
import pickle




if __name__=="__main__":

    
    books_overview = pickle.load(open(PATH_DATA + f'/remaining_book_titles.pkl', 'rb'))
    final_embeddings = pickle.load(open(PATH_EMBEDDINGS+ f'/Spacy_300dim_embeddings_epoch_8.pkl', 'rb')) # Training-Results after 8 Epochs
    doc_embeddings = final_embeddings[:308,:]
    doc_lower_dim_embedding = TSNE(n_components=2).fit_transform(doc_embeddings)
    
    plt.figure(figsize=(18,15))
    
    for index, title in enumerate (books_overview.keys()):
        
        #if (index >10 and index <40):
        x = doc_lower_dim_embedding[index,0]
        y = doc_lower_dim_embedding[index,1]
        plt.scatter(x, y, marker='x', color='red')
        plt.text(x, y,index, fontsize=10)
        

