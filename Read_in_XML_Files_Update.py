from bs4 import BeautifulSoup
import urllib
import os
import requests
from CONSTANTS import *


def create_start_soup(http_link):
    """Extract all links to the books
    """
    html_page = urllib.request.urlopen(http_link)  # the start page with the list of all books
    html_page = html_page.read()
    soup_1 = BeautifulSoup(html_page, "lxml")

    return soup_1


def download_books(soup_of_books, path):
    """Download the books and write them to directory
    """
    os.chdir(path)  # Select Directory, where Books should be written into #

    list_with_links = []
    i = 0

    for url in soup_of_books.find_all('a', attrs={'class': 'aResultsHeader'}):
        link_snippet = url.get('href')

        download_link = "http://www.perseus.tufts.edu/hopper/dl" + link_snippet  # only works for our use-case
        doc_name = "book" + str(i)

        with open(doc_name + ".xml", 'wb') as file:
            file.write(requests.get(download_link).content)

        list_with_links.append(download_link)
        print("Print Book Nr." + str(i))
        i = i + 1

    return list_with_links


def get_only_book_titles(bsoup):
    """Extract the names of all books, after the xml-files are downloaded
    """
    title_names = []
    for url in bsoup.find_all('a', attrs={'class': 'aResultsHeader'}):
        title_names.append(url.text)  # the list collects the names of all books

    return title_names


if __name__ == "__main__":

    main_soup = create_start_soup("http://www.perseus.tufts.edu/hopper/collection?collection=Perseus:collection:cwar")

    # Download the books to directory(adjust to your system) --> Only necessary once!!! (otherwise comment the line out)
    #download_links = download_books(main_soup, PATH_XML)
    ##

    title_names = get_only_book_titles(main_soup)
