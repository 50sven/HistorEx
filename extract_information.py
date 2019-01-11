from bs4 import BeautifulSoup
from collections import Counter
import spacy


def get_persons(file, num=10):
    """Extract most frequent persons from xml files by tag

    Args:
        file (string): path to .xml file
        num (int): number of persons returned

    Returns:
        persons_list (list): list of most common persons
        frequency_list (list): frequency of persons_list
    """
    soup = BeautifulSoup(open(file, "rb"), "html.parser")

    # Get text and authname of all persons if available
    persons = [(p.text, p["authname"]) for p in soup.find_all("persname") if "authname" in p.attrs]

    # Separate authnames and count the most common
    # (added half of num parameter, due to next step)
    authnames = [p[1] for p in persons]
    counter_authnames = Counter(authnames).most_common(2 * num)

    # For each authname list the names appearing in the raw text.
    # Throw out names without whitespaces and
    # count the number of whitespaces for the remaining ones.
    # (more whitespaces -> more likely to get the full name or more information)
    names_count = [
        (
            list(set([(pers[0], pers[0].count(" "))
                      for pers in persons
                      if (p[0] == pers[1] and " " in pers[0])])),
            p[1]
        )
        for p in counter_authnames
    ]

    # Exclude the raw text names including the number of whitesapces
    # and exclude the frequencies
    names_count = [n for n in names_count if n[0]]
    frequency_list = [n[1] for n in names_count]

    # Create a list of persons by sorting by number of whitespaces and length of name
    persons_list = [n[0] for n in names_count]
    persons_list = [sorted(p, key=lambda x: (x[1], len(x[0])),
                           reverse=True) for p in persons_list]
    persons_list = [p[0][0] for p in persons_list]

    # Finally, slice the most common persons with their frequencies
    persons_list = persons_list[::-1][-num:]
    frequency_list = frequency_list[::-1][-num:]

    return persons_list, frequency_list


def get_places(file, num=10):
    """Extract most frequent places from xml files by tag

    Args:
        file (string): path to .xml file
        num (int): number of places returned

    Returns:
        places_list (list): list of most common places
        frequency_list (list): frequency of places_list
    """
    soup = BeautifulSoup(open(file, "rb"), "html.parser")

    # Get text and authname of all places if available
    places = [(p["reg"], p["authname"]) for p in soup.find_all("placename") if ("authname" in p.attrs and "reg" in p.attrs)]

    # Separate authnames and count the most common
    authnames = [p[1] for p in places]
    counter_authnames = Counter(authnames).most_common(num)

    # For each authname list the reg names appearing in the tag
    # and exclude frequencies
    names_count = [
        (
            list(set([place[0] for place in places if p[0] == place[1]])),
            p[1]
        )
        for p in counter_authnames
    ]
    frequency_list = [n[1] for n in names_count]

    # Create a list of places by sorting by the longest name with capital letter
    places_list = [n[0] for n in names_count]
    places_list = [[n for n in p if n] for p in places_list]
    places_list = [sorted(p, key=lambda x: (len(x), x[0].isupper()),
                          reverse=True) for p in places_list]
    places_list = [p[0] for p in places_list]
    places_list = places_list[::-1]
    frequency_list = frequency_list[::-1]

    return places_list, frequency_list


def get_persons_and_places(file, num_persons=10, num_places=10):
    """Extract most frequent persons and places from xml files by tag

    COMBINES get_persons() and get_places()
    -> DETAILED INFORMATION IN THE DOCSTRINGS OF EACH FUNCTION

    Args:
        file (string): path to .xml file
        num_persons (int): number of persons returned
        num_places (int): number of places returned

    Returns:
        persons_list (list): list of most common persons
        persons_frequency_list (list): frequency of persons_list
        places_list (list): list of most common places
        places_frequency_list (list): frequency of places_list
    """

    persons_list, persons_frequency_list = get_persons(file, num_persons)
    places_list, places_frequency_list = get_places(file, num_places)

    return (persons_list, persons_frequency_list,
            places_list, places_frequency_list)


def get_persons_and_places_by_spacy(file, num_persons=10, num_places=10):
    """Extract most frequent persons and places from raw text files with spacy

    Args:
        file (string): path to .xml file
        num_persons (int): number of persons returned
        num_places (int): number of places returned

    Returns:
        persons_list (list): list of most common persons
        persons_frequency_list (list): frequency of persons_list
        places_list (list): list of most common places
        places_frequency_list (list): frequency of places_list
    """
    nlp = spacy.load('en_core_web_sm')

    with open(file, encoding='utf-8') as f:
        text = f.read()

    if len(text) > 1e6:
        n = 1e5
        text_chunks = [nlp(text[i:i + int(n)]) for i in range(0, len(text), int(n))]
        persons = [e.text for tc in text_chunks for e in tc.ents
                   if e.label_ == "PERSON"]
        places = [e.text for tc in text_chunks for e in tc.ents
                  if e.label_ == "GPE"]
    else:
        doc = nlp(text)
        persons = [e.text for e in doc.ents
                   if e.label_ == "PERSON"]
        places = [e.text for e in doc.ents
                  if e.label_ == "GPE"]

    persons = [p for p in persons if (" " in p and
                                      "\r" not in p and
                                      "\n" not in p)]
    places = [p for p in places if ("\r" not in p and
                                    "\n" not in p)]

    counter_persons = Counter(persons).most_common(num_persons)
    persons_list = [p[0] for p in counter_persons][::-1]
    persons_frequency_list = [p[1] for p in counter_persons][::-1]

    counter_places = Counter(places).most_common(num_places)
    places_list = [p[0] for p in counter_places][::-1]
    places_frequency_list = [p[1] for p in counter_places][::-1]

    return (persons_list, persons_frequency_list,
            places_list, places_frequency_list)


if __name__ == "__main__":

    import glob
    import natsort
    import pickle
    from CONSTANTS import *

    BOOK_TITLES = pickle.load(open("dashboard/assets/data_overview_tsne.pkl", "rb"))["titles"]

    files = glob.glob(PATH_XML + '/*')  # by tag
    # files = glob.glob(PATH_RAW_TEXT + '/*')  # by spacy
    files = natsort.natsorted(files)
    aggregated_data = dict()

    for idx, file in enumerate(files):
        print(f"Processing: {files[idx]}")

        persons_list, persons_frequency_list, places_list, places_frequency_list = get_persons_and_places(file, num_persons=10, num_places=20)
        # persons_list, persons_frequency_list, places_list, places_frequency_list = get_persons_and_places_by_spacy(file, num_persons=10, num_places=20)

        title = BOOK_TITLES[idx]

        aggregated_data[title] = {
            "persons": {
                "names": persons_list,
                "frequency": persons_frequency_list
            },
            "places": {
                "names": places_list,
                "frequency": places_frequency_list
            }
        }
        break
