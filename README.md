# Mental and Physical Health Dialogue and its effect on Perceived Social Support
This document's format is based off of https://github.com/behavioral-data/Empathy-Mental-Health/tree/master. Please note that portions of debugging were done with Microsoft Bing Search: Bing Chat Mode (2021,  Version 1.0) in all pieces of the code. All other sources used in the coding process are cited in comments of the wiki_page Python scripts unless the code was specifically for stripping paragraphs or sentences.

## Introduction
If the following commands do not work, try replacing 'python' with 'python3'.
### 1. Prerequisites

```
$ pip install -r requirements.txt
```

### 2. Prepare dataset
First, you will need to download the file 'communities_articles.csv' and have it in the same folder as 'wiki_collector.py'.
```
python wiki_collector.py
```
### 3. Training the models
Run the following to train Naive Bayes, SVM, Random Forest, Logistic Regression, and DistilBERT models on pages, paragraphs, and sentences.
```
python wiki_page_classifier.py
python wiki_page_distilbert.py 
python wiki_paragraph_classifier.py
python wiki_paragraph_distilbert.py 
python wiki_sentence_classifier.py
python wiki_sentence_distilbert.py 
```

# References
Microsoft Bing Search. (2021). Bing Chat Mode (Version 1.0) [Online chatbot]. https://www.bing.com/chat