
# coding: utf-8

# In[1]:


import os
import sys
import bs4
import requests
import time
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
import re
import csv
from fractions import Fraction
import builtins

def get_text_from_elements(elements):
    """Uses list comprehension to parse out the cleaned text strings from a list of
    elements returned from a BeautifulSoup selection.

    Arguments:
        elements {list} -- list of elements returned from a BeautifulSoup selection

    Returns:
        list -- list of cleaned text contained within the element list
    """
    
    text_list = [e.text.strip() for e in elements]
    if not (len(text_list) == 0):
        return text_list
    else:
        return 'NA'


# In[2]:


jour_urls = list()
for i in range(1,50,1):
    jour_urls.append("https://pubs.acs.org/toc/jacsat/139/"+ str(i))
#print(jour_urls)


# In[3]:


#put'em in a list
art_urls = list()

#Disabling JavaScript loading to speed up the process
chrome_options = Options()
chrome_options.add_experimental_option( "prefs",{'profile.managed_default_content_settings.javascript': 2})
browser = webdriver.Chrome('chromedriver',chrome_options=chrome_options)

browser = webdriver.Chrome()

for url in jour_urls:
    browser.get(url)
    #data = r.text
    time.sleep(5)
    soup = bs(browser.page_source, 'html.parser')
    for link in soup.find_all('a'):
    #print(link)
    #regular expression
        n = re.search('/doi/abs/', str(link.get('href')))
        if not pd.isnull(n):
        #print(str(link.get('href')))
            art_urls.append("https://pubs.acs.org"+str(link.get('href')))


# In[4]:


len(art_urls)


# In[5]:


#test run
sub_art_list = art_urls[0:len(art_urls)-1]

title_list = list()
sender_list = list()
receiver_list = list()
text_list = list()

title_list.append('Title')
sender_list.append('Sender')
receiver_list.append('Receiver')
text_list.append('Abstract')

#Disabling JavaScript loading to speed up the process
chrome_options = Options()
chrome_options.add_experimental_option( "prefs",{'profile.managed_default_content_settings.javascript': 2})
browser = webdriver.Chrome('chromedriver',chrome_options=chrome_options)

browser = webdriver.Chrome()

for url in sub_art_list:
    browser.get(url)
    #data = r.text
    time.sleep(5)
    soup = bs(browser.page_source, 'html.parser')
    title = get_text_from_elements(soup.select("h1.articleTitle"))[0]
    authors = get_text_from_elements(soup.select("a#authors"))
    textblock = get_text_from_elements(soup.select("p.articleBody_abstractText"))[0]
    if not (title is 'NA' or authors is 'NA' or textblock is 'NA'):
        title_list.append(title)
        sender_list.append(authors[0])
        receiver_list.append(authors[len(authors)-1])
        text_list.append(textblock)


# In[14]:


rows = zip(title_list,sender_list,receiver_list,text_list)

import csv
with open("list.csv", "w") as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)


# In[7]:


import csv
import json

csvfile = open('list.csv', 'r')
jsonfile = open('list.json', 'w')

fieldnames = ("Title","Sender","Receiver","Content")
reader = csv.DictReader(csvfile, fieldnames)
for row in reader:
    json.dump(row, jsonfile)
    jsonfile.write('\n')

