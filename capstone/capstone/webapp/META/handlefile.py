


import nltk
import json
import spacy
import re

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import random
from gensim.models import CoherenceModel
import gensim
from gensim import corpora,models
#@mallet_path = '/Users/Maggie/Downloads/mallet-2.0.8/bin/mallet' # update this path
from nltk.corpus import stopwords 
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.corpora import Dictionary
import string
#with open('total.json') as json_data:
#    content = json.load(json_data)

def handle_files(json_file):
# ### Author-Doc List

# In[24]:
content = json_file

# Get all author names and their corresponding document IDs.
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
    
i = 0    
for entry in content:
    receiver = entry['Receiver'].replace('\n',' ')
    if not author2doc.get(receiver):
        # This is a new author.
        author2doc[receiver] = []
        author2doc[receiver] = [i]
    # Add document IDs to author.
    else:
        author2doc[receiver].append(i)
    i = i + 1


# ### Clean text data

# In[25]:


nlp = spacy.load('en')

### using both title and abstract
abstract = []
for entry in content:
    title = entry['Title'].replace('\n',' ')
    title = title.replace('/u',' ')
    #sender = entry['Sender'].replace('\n',' ')
    #receiver = entry['Receiver'].replace('\n',' ')
    abst = entry['Content'].replace('\n',' ')
    abst = abst.replace('/u',' ')
    abst = abst.replace('%',' ')
    entry_str = title+' '+abst
    entry_str = re.sub(r'\b\w{1,3}\b', '',entry_str)
    abstract.append(entry_str)


# In[26]:


### Load stopwords

d = {}
stopword = stopwords.words('english')


# In[27]:


### lemmatization, bigrams
#%%time
processed_docs = []    
for doc in nlp.pipe(abstract, n_threads=4, batch_size=100):
    # Process document using Spacy NLP pipeline.
    
    ents = doc.ents  # Named entities.

    # Keep only words (no numbers, no punctuation).
    # Lemmatize tokens, remove punctuation and remove stopwords.
    doc = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    # Remove common words from a stopword list.
    doc = [token for token in doc if token not in stopword]

    # Add named entities, but only if they are a compound of more than word.
    doc.extend([str(entity) for entity in ents if len(entity) > 1])
    
    processed_docs.append(doc)

abstract_all = processed_docs
del processed_docs

from gensim.models import Phrases
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(abstract_all, min_count=20)
for idx in range(len(abstract_all)):
    for token in bigram[abstract_all[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            abstract_all[idx].append(token)



dictionary = Dictionary(abstract_all)

# Remove rare and common tokens.
# Filter out words that occur too frequently or too rarely.
max_freq = 0.2
min_wordcount = 80
dictionary.filter_extremes(no_below=min_wordcount, no_above=max_freq)

_ = dictionary[0]  # This sort of "initializes" dictionary.id2token.



Total = []
for c in content:
    ##using both title and content
    total = c['Title']
    Total.append(total)
    
stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(entry).split() for entry in Total]



def bestModel(abstract_all):
    co_score_tfidf = []
    co_score_lda = []
    co_score_mallet = []

    for i in range(0,10):
        random.shuffle(abstract_all)

        training = abstract_all[:round(len(abstract_all)*0.6)]
        test = abstract_all[round(len(abstract_all)*0.6):]

        doc_clean_train = [entry for entry in training]
        doc_clean_test = [entry for entry in test]
        # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
        dictionary_tr = corpora.Dictionary(doc_clean_train)
        dictionary_te = corpora.Dictionary(doc_clean_test)
        dictionary = corpora.Dictionary(abstract_all)
        # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
        doc_term_matrix_te = [dictionary_te.doc2bow(doc) for doc in doc_clean_test]
        doc_term_matrix_tr = [dictionary_tr.doc2bow(doc) for doc in doc_clean_train]
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in abstract_all]

        #mystring = mystring..decode(‘utf-8’)

        tfidf = models.TfidfModel(doc_term_matrix)
        corpus_tfidf = tfidf[doc_term_matrix_tr]
        corpus_tfidf_te = tfidf[doc_term_matrix_te]

        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary_tr, passes=2, workers=4)

        Lda = gensim.models.ldamodel.LdaModel
        ldamodel = Lda(doc_term_matrix_tr, num_topics=5, id2word = dictionary_tr, passes=50)
        ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=doc_term_matrix_tr, num_topics=20, id2word=dictionary_tr)
    
        #tfidf
        coherence_model_ldatfidf = CoherenceModel(model=lda_model_tfidf, texts=doc_clean_test, dictionary=dictionary_te, coherence='c_v')
        coherence_ldatfidf = coherence_model_ldatfidf.get_coherence()
    
        co_score_tfidf.append(coherence_ldatfidf)
    
        #lda
        coherence_model = CoherenceModel(model=ldamodel, texts=doc_clean_test, dictionary=dictionary_te, coherence='c_v')
        coherence_lda = coherence_model.get_coherence()
    
        co_score_lda.append(coherence_lda)
    
        #mallet
        coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=doc_clean_test, dictionary=dictionary_te, coherence='c_v')
        coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    
        co_score_mallet.append(coherence_ldamallet)
    
      
    
    avg_co_lda = sum(co_score_lda)/10
    avg_co_tfidf = sum(co_score_lda)/10
    avg_co_mallet = sum(co_score_mallet)/10

    result = {avg_co_lda:'lda',avg_co_tfidf:'tfidf',avg_co_mallet:'mallet'}
    maximum = max([avg_co_lda,avg_co_tfidf,avg_co_mallet])
    best = result[maximum]
    
    return best
    
    
    


# In[31]:


def compute_coherence_values(total, best, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    random.shuffle(total)

    training = total[:round(len(total)*0.6)]
    test = total[round(len(total)*0.6):]

    doc_clean_train = [clean(entry).split() for entry in training]
    doc_clean_test = [clean(entry).split() for entry in test]
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. 
    dictionary_tr = corpora.Dictionary(doc_clean_train)
    dictionary_te = corpora.Dictionary(doc_clean_test)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix_te = [dictionary_te.doc2bow(doc) for doc in doc_clean_test]
    doc_term_matrix_tr = [dictionary_te.doc2bow(doc) for doc in doc_clean_train]
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
     
    coherence_values = []
    model_list = []
    if(best == 'lda'):
        for n in range(start, limit, step):
            Lda = gensim.models.ldamodel.LdaModel
            ldamodel = Lda(doc_term_matrix_tr, num_topics=n, id2word = dictionary_tr, passes=50)
            coherence_model = CoherenceModel(model=ldamodel, texts=doc_clean_test, dictionary=dictionary_te, coherence='c_v')
            coherence_lda = coherence_model.get_coherence()
    
            coherence_values.append(coherence_lda)
         
    if(best == 'tfidf'):
        for n in range(start, limit, step):
            tfidf = models.TfidfModel(doc_term_matrix)
            corpus_tfidf = tfidf[doc_term_matrix_tr]
            corpus_tfidf_te = tfidf[doc_term_matrix_te]

            lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=n, id2word=dictionary_tr, passes=2, workers=4)
            coherence_model_ldatfidf = CoherenceModel(model=lda_model_tfidf, texts=doc_clean_test, dictionary=dictionary_te, coherence='c_v')
            coherence_ldatfidf = coherence_model_ldatfidf.get_coherence()
    
            coherence_values.append(coherence_ldatfidf)
   
    if(best == 'mallet'):
        for n in range(start, limit, step):
            ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=doc_term_matrix_tr, num_topics=n, id2word=dictionary_tr)
            coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=doc_clean_test, dictionary=dictionary_te, coherence='c_v')
            coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    
            coherence_values.append(coherence_ldamallet)
            

    return coherence_values





# In[32]:


def getOptimal(start,limit,step,coherence):
    x = range(start, limit, step)
    xlist = []
    for i, cv in zip(x,coherence):
        #print("Num Topics =", i, " has Coherence Value of", round(cv, 4))
        xlist.append(i)
    
    optimal = []
    last_x = start
    last_y = coherence[0]
    #last_slope = 1
    for i,cv in enumerate(coherence):
        #print("Num Topics =", m, " has Coherence Value of", round(cv, 4))
        last_slope = (cv-last_y)/step
        if i < len(coherence) - 1:
            next_y = coherence[i + 1]
            next_slope = (next_y-cv)/step
            if next_slope <= last_slope and next_slope >= 0  and i > 0:
                optimal.append((xlist[i]))
        else:
            break
        last_y = cv
        #last_x = i
    return min(optimal)

best = bestModel(abstract_all)
coherence = compute_coherence_values(Total, best, limit=40, start=2, step=6)
optimal_topics = getOptimal(2,40,6,coherence)





dictionary = corpora.Dictionary(abstract_all)


### AT Corpus
atcorpus = [dictionary.doc2bow(doc) for doc in abstract_all]

### LDA Mallet Corpus
from gensim.test.utils import datapath, get_tmpfile, common_texts
from gensim.corpora import MalletCorpus
from gensim.corpora import Dictionary

# Write corpus in Mallet format to disk
output_fname = get_tmpfile("corpus.mallet")
MalletCorpus.serialize(output_fname, atcorpus, dictionary)

mallet_corpus = MalletCorpus(output_fname)

malcorpus = list()

for t in mallet_corpus:
    malcorpus.append(t)

### LDA-tfidf Corpus
from operator import itemgetter
import gensim
from gensim import corpora,models
tfidf = models.TfidfModel(atcorpus)
corpus_tfidf = tfidf[atcorpus]

l = list()
for t in corpus_tfidf:
    l.append(t)

index = 0
tfidfcorpus = []
for i in l:
    index +=1
    common_denom = min(i,key=itemgetter(1))[1] if i else None
    if common_denom is not None:
        new_list = []
        for f in i:
            n = f[1]/common_denom
            new_list.append((f[0],int(n)))
        tfidfcorpus.append(new_list)
    else:
        #print(index)
        new_list = []
        for f in i:
            new_list.append(f[0],f[1])
        tfidfcorpus.append(new_list)




def showTopics(model, num):
    topics = []
    i = 1
    for topic in model.show_topics(num_topics=num):
        words = []
        for word, prob in model.show_topic(topic[0]):
            words.append(word)
        print('Topic '+str(i)+': ')
        print(words[2]+' '+words[1]+' '+words[0])
        print(*words)
        print()
        i += 1
        topics.append(words[2]+' '+words[1]+' '+words[0])
    return topics


