import pandas as pd
import requests
from wikidata.client import Client
import time
import datetime

# Link each author of a book with its wikidata page
start = time.time()

df = pd.read_csv('../data/1126_books_info.csv')
index = 0
url = "https://www.wikidata.org/w/api.php"
no_identifier = 1

for author in df['Author']:
    # add wikidata identifier
    if ';' in author:
        author = author.split(';')[0]
    params = {
        'action': 'wbsearchentities',
        'format': 'json',
        'language': 'en',
        'search': author
    }
    r = requests.get(url, params=params)
    if r.json()['search']:
        wiki_id = r.json()['search'][0]['id']
        df.loc[index, 'Author Wiki id'] = wiki_id
        df.loc[index, 'Author Wikidata Link'] = 'www.wikidata.org/wiki/' + wiki_id
        print(f'Wiki identifier: {wiki_id} of author {author} has been successfully retrieved. Index: {index}')
    else:
        print(f'No Identifier for {author} - count {no_identifier}')
        no_identifier += 1
    index += 1
    time.sleep(31)

df.to_csv('../data/' + str(datetime.datetime.now().month) + str(datetime.datetime.now().day) + '_books_info_wikiid.csv',
          index=False)
print(f'In Total {no_identifier} could not been found in Wikidata')
print(f'Time Tracking: in total it took {time.time()-start}')
#TODO add set with author who could not be found on wikidata

# ---- request to get id of an author
# url = "https://www.wikidata.org/w/api.php"
# query = 'Thomas Wentworth Higginson'
# params = {
#     'action': 'wbsearchentities',
#     'format': 'json',
#     'language': 'en',
#     'search': query
# }
# r = requests.get(url, params= params)
# id = r.json()['search'][0]['id']
# print(id)


# ---- get page of author using wikidata library
# client = Client()
# author = client.get(id)
# print(author)
# print(author.description)

# ---- request to  author page
# url2 = 'https://www.wikidata.org/w/api.php?action=parse&page=Q24871&format=json'
# response = requests.get(url2)
# print(response)
