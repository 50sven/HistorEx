# Extract Persons and Locations from each book

import glob
import re
from CONSTANTS import *
from bs4 import BeautifulSoup
import pandas as pd
import parsing
import datetime
import numpy as np
import time



def get_soup(file):
    soup = ""
    soup = BeautifulSoup(file, "html.parser")
    #persName = soup.body.find_all('persName')
    #print(persName)
    return soup


def extract_pers(f):
    name_list = []
    persName_re = re.findall(r'<persName .*?</persName>', f)
    if persName_re:
        for pers in persName_re:
            pers_name_re = re.search(r'n=\".*?\"', pers)
            pers_name = pers_name_re.group().split("\"")[1]
            pers_name = pers_name.replace(',', '')
            pers_name_table = re.sub(r"(\w)([A-Z])", r"\1, \2", pers_name)
            name_list.append(pers_name_table)
    return name_list


def extract_loc(f):
    location_list = []
    persLoc_re = re.findall(r'<placeName .*?</placeName>', f)
    if persLoc_re:
        for loc in persLoc_re:
            loc_name_re = re.search(r'>.*?<', loc)
            loc_name = loc_name_re.group().replace('<','>').split('>')[1]
            location_list.append(loc_name)
    return location_list


def write_df(df, df_index, element_list, nb, booktitle):#, column_list):
    #df = pd.concat([df, pd.DataFrame(columns=column_list)])
    df = df.append({'Booknb': nb, 'Title': booktitle}, ignore_index=True)
    for element in element_list:
        existing_element = list(df)
        if element in existing_element:
            # plus 1
            df.loc[df_index, element] += 1
        else:
            df.loc[df_index, element] = 1
    return df


def get_columns(df, column_list):
    column_list = column_list.append(list(df))
    return column_list


start = time.time()

files = glob.glob(PATH_XML + '/*')

# SET WINDOWS_OS
windows_OS = True
name_df = pd.DataFrame(columns=['Booknb', 'Title'])
location_df = pd.DataFrame(columns=['Booknb', 'Title'])
name_df_columns = []
locations_df_columns = []
for idx, file in enumerate(files):
    #name_df = pd.DataFrame(columns=['Booknb', 'Title'])
    #location_df = pd.DataFrame(columns=['Booknb', 'Title'])
    print(f"Processing: {files[idx]}")
    book_number = re.findall("\d+", file)[1]

    with open(file, "rb") as f:
        if (windows_OS):
            f = f.read()
            f = f.decode('utf-8', 'ignore').encode('latin-1', 'ignore').decode('utf-8', 'ignore')
        file_soup = get_soup(f)


        # get title author, date
        title, author, date = parsing.get_meta_data(file_soup)
        # get mentioned names of the book
        names = extract_pers(f)
        if names:
            name_df = write_df(name_df, idx, names, book_number, title)#, name_df_columns)
        else:
            name_df = name_df.append({'Booknb': book_number, 'Title': title}, ignore_index=True)
        # get mentioned locations of the book
        locations = extract_loc(f)
        if locations:
            location_df = write_df(location_df, idx, locations, book_number, title)#, name_df_columns)
        else:
            location_df = location_df.append({'Booknb': book_number, 'Title': title}, ignore_index=True)
        #name_df_columns = get_columns(name_df, name_df_columns)
        #locations_df_columns = get_columns(location_df, locations_df_columns)
        print(f'Name Dataframe Dim: {name_df.shape}')
        print(f'Location Dataframe Dim: {location_df.shape}')
name_df.to_csv('../data/' + str(datetime.datetime.now().month) + str(datetime.datetime.now().day) + '_books_names.csv', index=False)
location_df.to_csv('../data/' + str(datetime.datetime.now().month) + str(datetime.datetime.now().day) + '_books_locations.csv', index=False)
print(f'Total Time: {time.time() - start}')

