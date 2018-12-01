
# coding: utf-8

# In[1]:


def readJSON(fileName):
    import json
    import re

    content = []

    content = []
    with open(fileName) as json_data:
        content = json.load(json_data)
        
    author2doc = dict()

    i = 0
    for entry in content:
        sender = entry['Sender'].replace('\n',' ')
        if not author2doc.get(sender):
        # This is a new author.
        #author2doc[sender] = []
            author2doc[sender] = [i]
    # Add document IDs to author.
        else:
            author2doc[sender].append(i)
        i = i + 1
    
    return author2doc


# In[5]:


def writeJSON(some_dict, fileName):
    import json
    d = []
    for author, doc in some_dict.items():
        d.append({"author": author,
           "doc_id":doc})
    jsonfile = open(fileName, 'w')
    json.dump(d, jsonfile)
    jsonfile.close()
    return fileName


# In[6]:


#writeJSON(readJSON('total.json'))

