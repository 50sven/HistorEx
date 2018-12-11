import pandas as pd
import requests
from wikidata.client import Client
import time
import datetime

# Link each author of a book with its wikidata page
start = time.time()

def check_human(id):
    url = 'https://query.wikidata.org/sparql'
    query = 'PREFIX entity: <http://www.wikidata.org/entity/> ' \
            'SELECT ?human ' \
            'WHERE { entity:' + id + ' wdt:P31 ?human. ' \
                                     '} '
    response = requests.get(url, params={'format': 'json', 'query': query})
    data = response.json()
    if data['results']['bindings']:
        if 'Q5' in data['results']['bindings'][0]['human']['value']:
            human = True
        else:
            human = False
    return human



df = pd.read_csv('../data/1210_books_info.csv')
index = 0
url = "https://www.wikidata.org/w/api.php"
no_identifier = 1
no_author_wiki_id = set()
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
    response = r.json()
    if response['search']:
        wiki_id = response['search'][0]['id']
        #TODO get description
        human = check_human(wiki_id)
        if human:
            df.loc[index, 'author_wikidata_id'] = wiki_id
            df.loc[index, 'author_wikidata_link'] = 'www.wikidata.org/wiki/' + wiki_id
        print(f'Wiki identifier: {wiki_id} of author {author} has been successfully retrieved. Index: {index}')
    else:
        print(f'No Identifier for {author} - count {no_identifier}')
        no_identifier += 1
        no_author_wiki_id.add(author)
        print(f'No Wiki Id for: {no_author_wiki_id}')
    index += 1
    time.sleep(32)


df.to_csv('../data/' + str(datetime.datetime.now().month) + str(datetime.datetime.now().day) + '_books_info_wikiid.csv',
          index=False, sep='|')
with open('../data/' + str(datetime.datetime.now().month) + str(datetime.datetime.now().day) + '_author_set.txt', 'w+')as author_set:
    author_set.write(str(no_author_wiki_id))
print(f'In Total {no_identifier} could not been found in Wikidata')
print(f'Time Tracking: in total it took {time.time()-start}')

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
