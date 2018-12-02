from bs4 import BeautifulSoup
import re
from CONSTANTS import *
import glob
import parsing
import pandas as pd
import datetime

def get_xml(file, path, return_text=False, windows_OS=True):
    with open(file, "rb") as f:
        if (windows_OS):
            f = f.read()
            f = f.decode('utf-8', 'ignore').encode('latin-1', 'ignore').decode('utf-8', 'ignore')

        soup = BeautifulSoup(f, "html.parser")
    return soup

files = glob.glob(PATH_XML + '/*')
df = pd.DataFrame(columns=['Booknb', 'Title', 'Author', 'Publishing Date'])

for idx, file in enumerate(files):
    print(f"Processing: {files[idx]}")
    book_number = re.findall("\d+", file)[1]
    html_soup = get_xml(file, PATH_RAW_TEXT + book_number + ".txt",
                        return_text=False, windows_OS=True)
    title, author, date = parsing.get_meta_data(html_soup)
    df = df.append({'Booknb': book_number, 'Title': title, 'Author': author, 'Publishing Date': date}, ignore_index=True)
    df.to_csv(PATH_DF + str(datetime.datetime.now().month) + str(datetime.datetime.now().day) + '_books_info.csv', index=False)

print(df)
