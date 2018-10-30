from bs4 import BeautifulSoup


def xml_to_text(file, path, return_text=False):
    """Extract the text body from an .xml file

    Args:
        file (string): path to .xml file
        path (string): path for saving .txt file
        return_text (boolean): True, if text body should be returned

    Returns:
        text body (string): return the extracted text body
    """
    with open(file) as f:
        soup = BeautifulSoup(f, "html.parser")
    with open(path, "w") as t:
        for idx, chapter in enumerate(soup.find_all("div1")):
            try:
                t.write("[CHAPTER]" + chapter.head.text + "\n\n")
            except:
                t.write("[CHAPTER] No Header")
            last_p = "last_word_sentence_or_section"
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

    files = glob.glob('data/books/*')
    random_file = np.random.choice(files)

    for file in files:
        x = xml_to_text(file, "data/test.txt", return_text=True)
        break
