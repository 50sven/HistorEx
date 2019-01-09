from bs4 import BeautifulSoup
from collections import Counter
import pickle
from CONSTANTS import *

BOOK_TITLES = pickle.load(open("dashboard/assets/data_overview_tsne.pkl", "rb"))["titles"]


def get_persons(file, num=10):
    """Extract most frequent persons from xml files by tag

    Args:
        file (string): path to .xml file

    Returns:
        data (dictionary): return the extracted persons
    """
    soup = BeautifulSoup(open(file, "rb"), "html.parser")

    # Look for all tagged person names
    persons = [p.text for p in soup.find_all("persname") if " " in p.text]
    counter = Counter(persons)
    most_frequent_persons = counter.most_common(num)
    person_list = [p[0] for p in most_frequent_persons][::-1]
    frequency_list = [p[1] for p in most_frequent_persons][::-1]

    return person_list, frequency_list


def get_places(file, num=10):
    """Extract most frequent places from xml files by tag

    Args:
        file (string): path to .xml file

    Returns:
        data (dictionary): return the extracted places
    """
    soup = BeautifulSoup(open(file, "rb"), "html.parser")

    # Look for all tagged place names
    places = [p.text for p in soup.find_all("placename")]
    counter = Counter(places)
    most_frequent_places = counter.most_common(num)
    place_list = [p[0] for p in most_frequent_places][::-1]
    frequency_list = [p[1] for p in most_frequent_places][::-1]

    return place_list, frequency_list


def get_persons_and_places(file, num_person=10, num_places=10):
    """Extract most frequent persons and places from xml files by tag

    Args:
        file (string): path to .xml file

    Returns:
        data (dictionary): return the extracted persons and places
    """
    soup = BeautifulSoup(open(file, "rb"), "html.parser")

    # Look for all tagged person and place names
    persons = [p.text for p in soup.find_all("persname") if " " in p.text]
    counter = Counter(persons)
    most_frequent_persons = counter.most_common(num_person)
    person_list = [p[0] for p in most_frequent_persons][::-1]
    person_frequency_list = [p[1] for p in most_frequent_persons][::-1]

    places = [p.text for p in soup.find_all("placename")]
    counter = Counter(places)
    most_frequent_places = counter.most_common(num_places)
    place_list = [p[0] for p in most_frequent_places][::-1]
    place_frequency_list = [p[1] for p in most_frequent_places][::-1]

    return person_list, person_frequency_list, place_list, place_frequency_list


if __name__ == "__main__":

    import glob
    import natsort

    files = glob.glob(PATH_XML + '/*')
    files = natsort.natsorted(files)

    aggregated_data = dict()

    for idx, file in enumerate(files):
        print(f"Processing: {files[idx]}")
        person_list, person_frequency_list, place_list, place_frequency_list = get_persons_and_places(file)

        title = BOOK_TITLES[idx]

        aggregated_data[title] = {
            "persons": {
                "names": person_list,
                "frequency": person_frequency_list
            },
            "places": {
                "names": place_list,
                "frequency": place_frequency_list
            }
        }
