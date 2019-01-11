from bs4 import BeautifulSoup
import re
from CONSTANTS import *


def get_meta_data(soup):
    """Extract meta data from a .xml file soup
    """
    # title statement contains title and author
    # sourcDesc often contains publishing date
    titleStmt = soup.find("titlestmt")
    sourceDesc = soup.find("sourcedesc")
    title = ""
    author = ""
    date = 0

    # Look for all title snippets and concatenate them
    for idx, t in enumerate(titleStmt.find_all("title")):
        if idx == 0:
            title += t.text
        else:
            title += ", " + t.text

    # Look for author(s)
    for idx, a in enumerate(titleStmt.find_all("author")):
        if idx == 0:
            author += a.text
        else:
            author += "; " + a.text

    # Look for editor(s)
    if not author:
        for idx, e in enumerate(titleStmt.find_all("editor")):
            if idx == 0:
                author += e.text
            else:
                author += "; " + e.text

    # Look for society name
    if not author:
        fileDesc = soup.find("filedesc")
        try:
            orgname = fileDesc.find("orgname").text
            if orgname in title:
                author = "Medford Historical Society"
            else:
                author = "Somerville Historical Society"
        except:
            pass

    # Look for publishing date in source description or publishing statement
    try:
        date = re.findall(r'\d{4}', sourceDesc.text)[-1]
    except:
        pass
    if not date:
        publicationStmt = soup.find("publicationstmt")
        date = re.findall(r'\d{4}', publicationStmt.text)[-1]

    # Adapt results
    if not title:
        title = "-"
    if not author:
        author = "-"
    if not date:
        date = "-"

    return title, author, date


def xml_to_text(file, path, return_text=False, windows_OS=True):
    """Extract meta information and text body from an .xml file

    Args:
        file (string): path to .xml file
        path (string): path for saving .txt file
        return_text (boolean): True, if text body should be returned

    Returns:
        text (string): return the extracted meta information and text body
    """
    with open(file, "rb") as f:

        if(windows_OS):
            f = f.read()
            f = f.decode('utf-8', 'ignore').encode('latin-1', 'ignore').decode('utf-8', 'ignore')

        soup = BeautifulSoup(f, "html.parser")

    with open(path, "w") as t:

        # Extract meta data
        title, author, date = get_meta_data(soup)
        t.write("[TITLE] " + title + "\n")
        t.write("[AUTHOR] " + author + "\n")
        t.write("[DATE] " + date + "\n\n")

        # Go through each chapter and extract the text
        for idx, chapter in enumerate(soup.find_all("div1")):
            try:
                t.write("[CHAPTER] " + chapter.head.text + "\n\n")
            except:
                t.write("[CHAPTER] No Header" + "\n\n")
            last_p = "last_word_sentence_or_section"

            # Handle line breaks
            for i, p in enumerate(chapter.find_all("p")):
                text = p.text.strip().replace("\n", " ").replace("  ", " ") + "\n"
                if last_p == text:
                    continue
                last_p = text
                t.write(text)
            t.write("\n")

    if return_text:
        with open(path) as t:
            return t.read()


if __name__ == "__main__":

    import glob
    import numpy as np

    files = glob.glob(PATH_XML+'/*')

    for idx, file in enumerate(files):
        print(f"Processing: {files[idx]}")
        book_number = re.findall("\d+", file)[0]
        x = xml_to_text(file, PATH_RAW_TEXT+f"/book{book_number}.txt",
                        return_text=False,windows_OS=True)

    # random_file = np.random.choice(files)
    # print("Processing:", random_file)
    # x = xml_to_text(random_file, "data/test.txt", return_text=False)
