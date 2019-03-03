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
url = 'https://query.wikidata.org/sparql'
no_identifier = 1
no_author_wiki_id = set()

item_counter = 0
country_counter = 0
birth_counter = 0
death_counter = 0
birthplace_counter = 0
gender_counter = 0
occupation_counter = 0
image_counter = 0

for author in df['Author']:
    # add wikidata identifier
    if ';' in author:
        author = author.split(';')[0]
    #  https://query.wikidata.org/#SELECT%20distinct%20%3FitemLabel%20%3FitemDescription%20%3Fcountry%20%0A%3Fdate_birth%20%3Fdate_death%20%3Fbirth_place%20%3Fgender%20%3Foccupation%0AWHERE%20%0A%7B%0A%20%20%3FitemLabel%20%3Flabel%20%27John%20G.%20B.%20Adams%27%40en.%0A%20%20%3FitemLabel%20wdt%3AP31%20wd%3AQ5%20.%0A%20%20OPTIONAL%20%7B%20%3FitemLabel%20wdt%3AP21%20%3Fgender%20.%7D%0A%20%20OPTIONAL%20%7B%20%3FitemLabel%20wdt%3AP27%20%3Fcountry%20.%7D%0A%20%20OPTIONAL%20%7B%20%3FitemLabel%20wdt%3AP569%20%3Fdate_birth%20.%7D%0A%20%20OPTIONAL%20%7B%20%3FitemLabel%20wdt%3AP570%20%3Fdate_death%20.%7D%0A%20%20OPTIONAL%20%7B%20%3FitemLabel%20wdt%3AP19%20%3Fbirth_place%20.%7D%0A%20%20OPTIONAL%20%7B%20%3FitemLabel%20wdt%3AP106%20%3Foccupation%20.%7D%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%0A%09%09bd%3AserviceParam%20wikibase%3Alanguage%20%22en%22%20.%0A%09%7D%0A%20%20%7D
    #  https://query.wikidata.org/#SELECT%20distinct%20%3Fitem%20%3FitemLabel%20%3FitemDescription%20%3FcountryLabel%20%0A%3Fdate_birth%20%3Fdate_death%20%3Fbirth_placeLabel%20%3FgenderLabel%20%3FoccupationLabel%20%3Fimage%0AWHERE%20%0A%7B%0A%20%20%3Fitem%20%3Flabel%20%27Elizabeth%20Cary%20Agassiz%27%40en.%0A%20%20%3Fitem%20wdt%3AP31%20wd%3AQ5%20.%0A%20%20OPTIONAL%20%7B%20%3Fitem%20wdt%3AP21%20%3Fgender%20.%7D%0A%20%20OPTIONAL%20%7B%20%3Fitem%20wdt%3AP27%20%3Fcountry%20.%7D%0A%20%20OPTIONAL%20%7B%20%3Fitem%20wdt%3AP569%20%3Fdate_birth%20.%7D%0A%20%20OPTIONAL%20%7B%20%3Fitem%20wdt%3AP570%20%3Fdate_death%20.%7D%0A%20%20OPTIONAL%20%7B%20%3Fitem%20wdt%3AP19%20%3Fbirth_place%20.%7D%0A%20%20OPTIONAL%20%7B%20%3Fitem%20wdt%3AP106%20%3Foccupation%20.%7D%0A%20%20OPTIONAL%20%7B%20%3Fitem%20wdt%3AP18%20%3Fimage%20.%7D%0A%20%20SERVICE%20wikibase%3Alabel%20%7B%0A%09%09bd%3AserviceParam%20wikibase%3Alanguage%20%22en%22%20.%0A%09%7D%0A%20%20%7D
    sparql_query = 'SELECT distinct ?item ?itemLabel ?itemDescription ?countryLabel ?date_birth ?date_death ' \
                   '?birth_placeLabel ?genderLabel ?occupationLabel ?image ' \
                   'WHERE { ' \
                   '?item ?label \'' + author + '\'@en. ' \
                   '?item wdt:P31 wd:Q5. ' \
                    'OPTIONAL { ?item wdt:P21 ?gender .}' \
                    'OPTIONAL { ?item wdt:P27 ?country .} ' \
                    'OPTIONAL { ?item wdt:P569 ?date_birth .} ' \
                    'OPTIONAL { ?item wdt:P570 ?date_death .} ' \
                    'OPTIONAL { ?item wdt:P19 ?birth_place .} ' \
                    'OPTIONAL { ?item wdt:P106 ?occupation .}' \
                    'OPTIONAL { ?item wdt:P18 ?image .}' \
                   'SERVICE wikibase:label { ' \
                        'bd:serviceParam wikibase:language "en". ' \
                    '}' \
                   '}'
    r = requests.get(url, params={'format': 'json', 'query': sparql_query})
    response = r.json()
    if response['results']['bindings']:
        occupation = ""
        for key in response['results']['bindings'][0]:
            if key == 'item':
                wikiid_binding0 = response['results']['bindings'][0]['item']['value']
                df.loc[index, 'author_wikidata_id'] = response['results']['bindings'][0]['item']['value']
                item_counter += 1
            elif key == 'itemLabel':
                df.loc[index, 'author'] = response['results']['bindings'][0]['itemLabel']['value']
            elif key == 'itemDescription':
                df.loc[index, 'author_description'] = response['results']['bindings'][0]['itemDescription']['value']
            elif key == 'genderLabel':
                df.loc[index, 'gender'] = response['results']['bindings'][0]['genderLabel']['value']
                gender_counter += 1
            elif key == 'countryLabel':
                df.loc[index, 'origin'] = response['results']['bindings'][0]['countryLabel']['value']
                country_counter += 1
            elif key == 'date_birth': # todo how are only years handled?
                birth_dt = datetime.datetime.strptime(response['results']['bindings'][0]['date_birth']['value'], '%Y-%m-%dT%H:%M:%SZ')
                df.loc[index, 'date_birth'] = f'{birth_dt.day}/{birth_dt.month}/{birth_dt.year}'
                birth_counter += 1
            elif key == 'birth_placeLabel':
                df.loc[index, 'birth_place'] = response['results']['bindings'][0]['birth_placeLabel']['value']
                birthplace_counter += 1
            elif key == 'date_death':
                death_dt = datetime.datetime.strptime(response['results']['bindings'][0]['date_death']['value'],
                                                      '%Y-%m-%dT%H:%M:%SZ')
                df.loc[index, 'date_death'] = f'{death_dt.day}/{death_dt.month}/{death_dt.year}'
                death_counter += 1
            elif key == 'occupationLabel':
                df.loc[index, 'occupation'] = response['results']['bindings'][0]['occupationLabel']['value']
                occupation_counter += 1
            elif key == 'image':
                df.loc[index, 'author_image'] = response['results']['bindings'][0]['image']['value']
                image_counter += 1

    len_bindings = len(response['results']['bindings'])
    counter = 0
    if len_bindings > 1:
        occupation = ""
        for occup_result in response['results']['bindings']:
            if counter == len_bindings:
                break
            if 'occupationLabel' in response['results']['bindings'][counter].keys() and response['results']['bindings'][counter]['item']['value'] == wikiid_binding0:
                if occupation: # todo string als liste
                    occupation += ', ' + response['results']['bindings'][counter]['occupationLabel']['value']
                else:
                    occupation += response['results']['bindings'][counter]['occupationLabel']['value']
                counter += 1
        df.loc[index, 'occupation'] = str(occupation)
    index += 1
    print(f'--- Processed author: {author} ; Index: {index}')
    print(f'{item_counter / index * 100} % items were found on wikidata.')
    print(f'{gender_counter / index * 100} % gender information were found on wikidata.')
    print(f'{country_counter / index * 100} % origin information were found on wikidata.')
    print(f'{birth_counter / index * 100} % birth information were found on wikidata.')
    print(f'{birthplace_counter / index * 100} % birthplace information were found on wikidata.')
    print(f'{death_counter / index * 100} % death information were found on wikidata.')
    print(f'{occupation_counter / index * 100} % occupation information were found on wikidata.')
    print(f'{image_counter / index * 100} % images were found on wikidata.')
    time.sleep(33)


df.to_csv('../data/' + str(datetime.datetime.now().month) + str(datetime.datetime.now().day) + '_books_info_wiki.csv',
          index=False, sep='|')
print(f'Time Tracking: in total it took {time.time()-start}')
