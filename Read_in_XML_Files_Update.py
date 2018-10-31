# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 14:00:56 2018

@author: Michael
"""

from bs4 import BeautifulSoup, SoupStrainer
import urllib
import re
import time
import os
import requests
import pickle
import sys
import xml.etree.ElementTree as ET 
from xml.dom import minidom



"""
extract all links to the text
"""
html_page = urllib.request.urlopen("http://www.perseus.tufts.edu/hopper/collection?collection=Perseus:collection:cwar") # the start page with the list of all books
html_page = html_page.read()
soup_1 = BeautifulSoup(html_page, "lxml")


# Select Directory, where Books should be written into #
os.chdir("C:/Users/Michael/Documents/KIT/Information_Service_Engineering/Books_new")


# Download the books and write them to directory #

list_with_links = []
list_with_file_names = []
i = 0

for url in soup_1.find_all('a', attrs={'class' : 'aResultsHeader'}):
    link_snippet = url.get('href')
    
    download_link = "http://www.perseus.tufts.edu/hopper/dl"+link_snippet
    doc_name = "book"+ str(i)
    
    with open(doc_name+".xml", 'wb') as file:
        file.write (requests.get(download_link).content)
        
    
    list_with_links.append(download_link)
    list_with_file_names.append(doc_name)
    
    print(i)
    i = i+1
    


# Extract the names of all books, after the xml-files are downloaded #
title_names = []
for url in soup_1.find_all('a', attrs={'class' : 'aResultsHeader'}):
    title_names.append(url.text) # the list collects the names of all books


# Save und reload the names -------------> Adjust the Paths #
pickle.dump(title_names, open('C:/Users/Michael/Documents/KIT/Information_Service_Engineering/Book_Titles/Book_Titles.pkl', 'wb')) 
Title_Names = pickle.load(open('C:/Users/Michael/Documents/KIT/Information_Service_Engineering/Book_Titles/Book_Titles.pkl', 'rb'))   
##    
    
    
