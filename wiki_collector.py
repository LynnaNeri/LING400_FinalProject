# https://levelup.gitconnected.com/two-simple-ways-to-scrape-text-from-wikipedia-in-python-9ce07426579b
# https://stackoverflow.com/questions/76187256/importerror-urllib3-v2-0-only-supports-openssl-1-1-1-currently-the-ssl-modu
import os
import pandas as pd
import requests
import bs4
import re
import glob
from pathlib import Path

class wiki_collector:
    def __init__(self):
        self.rows = []

    def make_directory(self): #https://www.geeksforgeeks.org/create-a-directory-in-python/#
        directory = 'wiki_pages'
        parent_directory = os.getcwd() #https://www.geeksforgeeks.org/find-path-to-the-given-file-using-python/
        path = os.path.join(parent_directory, directory)
        pathexists = os.path.exists(path) #https://www.geeksforgeeks.org/python-check-if-a-file-or-directory-exists-2/
        if not pathexists:
            os.mkdir(path)
        os.chdir('wiki_pages') #https://www.geeksforgeeks.org/python-os-chdir-method/
        return str(path)
    
    def make_directory_mental(self): #https://www.geeksforgeeks.org/create-a-directory-in-python/#
        directory = 'mental_pages'
        parent_directory =  os.getcwd()
        path = os.path.join(parent_directory, directory)
        pathexists = os.path.exists(path) #https://www.geeksforgeeks.org/python-check-if-a-file-or-directory-exists-2/
        if not pathexists:
            os.mkdir(path)
        return str(path)
    
    def make_directory_physical(self): #https://www.geeksforgeeks.org/create-a-directory-in-python/#
        directory = 'physical_pages'
        parent_directory =  os.getcwd()
        path = os.path.join(parent_directory, directory)
        pathexists = os.path.exists(path) #https://www.geeksforgeeks.org/python-check-if-a-file-or-directory-exists-2/
        if not pathexists:
            os.mkdir(path)
        return str(path)

    def read_csv(self):
        #Reading in the csv file.
            #https://www.geeksforgeeks.org/working-csv-files-python/
        with open('../communities_articles.csv', 'r') as csvfile: 
            csvfile = pd.read_csv(csvfile) #https://stackoverflow.com/questions/23748995/pandas-dataframe-column-to-list
        csvfile = csvfile.dropna()
        csvfile['Url_Type'] = list(zip(csvfile['Wikipedia_URL'], csvfile['Type_by_ICD-11'])) #https://stackoverflow.com/questions/16031056/how-to-form-tuple-column-from-two-columns-in-pandas
        page_url_type = csvfile['Url_Type'].to_list()
        return page_url_type
    
    def wiki_extractor(self):
        pages = self.read_csv()
        for page in pages: #https://stackoverflow.com/questions/61837649/extracting-data-from-wikipedia-to-a-txt-file-using-python#:~:text=wiki_page%20%3D%20%27Agriculture%27%20res%20%3D%20requests.get%20%28f%27https%3A%2F%2Fen.wikipedia.org%2Fwiki%2F%20%7Bwiki_page%7D%27%29,each%20paragraph%20to%20the%20file%20f.write%20%28i.getText%20%28%29%29
            res = requests.get(page[0])
            res.raise_for_status()
            wiki = bs4.BeautifulSoup(res.text,"html.parser")

            # open a file named as your wiki page in write mode
            page_name = re.search('.*wiki\/(.*)',page[0]).group(1) #https://regex101.com/, https://stackoverflow.com/questions/24280607/capturing-subset-of-a-string-using-pythons-regex
            if page[1] == 'Mental':
                with open(self.make_directory_mental() + "/" + page_name + ".txt", "w", encoding="utf-8") as f: 
                    for i in wiki.select('p'):
                        # write each paragraph to the file
                        f.write(i.getText())
                #print('Mental: ', page_name)
            if page[1] == 'Physical':
                with open(self.make_directory_physical() + "/" + page_name + ".txt", "w", encoding="utf-8") as f: 
                    for i in wiki.select('p'):
                        # write each paragraph to the file
                        f.write(i.getText())
                #print('Physical: ', page_name)

    def test_collector(self):
    #https://stackoverflow.com/questions/17749058/combine-multiple-text-files-into-one-text-file-using-python
        proj_dir = os.getcwd()
        os.chdir('mental_pages')
        cur_dir = os.getcwd()
        read_files = glob.glob("*.txt")
        with open("mental_docs.txt", "wb") as outfile:
            for f in read_files:
                with open(f, "rb") as infile:
                    outfile.write(infile.read())
        Path(cur_dir + "/mental_docs.txt").rename(proj_dir + "/mental_docs.txt")
        os.chdir('../')
        print('Collected mental health test data.')
        
        os.chdir('physical_pages')
        cur_dir = os.getcwd()
        read_files = glob.glob("*.txt")
        with open("physical_docs.txt", "wb") as outfile:
            for f in read_files:
                with open(f, "rb") as infile:
                    outfile.write(infile.read())
        Path(cur_dir + "/physical_docs.txt").rename(proj_dir + "/physical_docs.txt")
        os.chdir('../')
        print('Collected physical health test data.')

if __name__ == '__main__':
    wiki_reader = wiki_collector()
    wiki_reader.make_directory()
    wiki_reader.make_directory_mental()
    wiki_reader.make_directory_physical()
    wiki_reader.read_csv()
    wiki_reader.wiki_extractor()
    wiki_reader.test_collector()