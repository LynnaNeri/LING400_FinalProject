#https://stackoverflow.com/a/31505798
import os
import nltk
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from joblib import dump

#DATASET BUILDING
print("*Building Dataset*")
#Collect pages related to mental health.
mental = []
mental_par_count = []
os.chdir('wiki_pages/mental_pages')
for file in os.listdir(): #https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
    if not file.startswith('.') and not file.startswith('Icon'):
        page = file
        cur_file = os.getcwd() + '/' + file
        text = Path(cur_file).read_text() #https://stackoverflow.com/questions/3758147/easiest-way-to-read-write-a-files-content-in-python
        text = text.split('\n')
        for contents in text:
            if contents != '' or '\s':
                mental.append((page, 0, contents))
    
    count = 0
    for paragraph in mental:
        if paragraph[0] == page:
            count += 1
    mental_par_count.append((page, count))
    
#Collect pages related to physical health.
physical = []
physical_par_count = []
os.chdir('../physical_pages')
for file in os.listdir():
    if not file.startswith('.') and not file.startswith('Icon'):
        page = file
        cur_file = os.getcwd() + '/' + file
        text = Path(cur_file).read_text()
        text = text.split('\n')
        for contents in text:
            if contents != '' or '\s':
                physical.append((page, 1, contents))
    
    count = 0
    for paragraph in physical:
        if paragraph[0] == page:
            count += 1
    physical_par_count.append((page, count))

paragraph_counts = []
for file in mental_par_count:
    paragraph_counts.append(file[1])
for file in physical_par_count:
    paragraph_counts.append(file[1])
minimum_paragraphs = min(paragraph_counts)

os.chdir('../mental_pages')
cut_mental = []
cut_mental_count = []
cut_mental_topics = []
for file in os.listdir():
    count = 0
    for paragraph in mental:
        if paragraph[0] == file:
            if count < minimum_paragraphs:
                cut_mental.append(paragraph)
                count += 1
    count = 0
    topic = []
    for paragraph in cut_mental:
        if paragraph[0] == file:
            count += 1
            topic.append(paragraph[1:])
    cut_mental_topics.append((file, topic))
    cut_mental_count.append((file, count))
#print(cut_mental_count)
#print(cut_mental_topics[1])

os.chdir('../physical_pages')
cut_physical = []
cut_physical_count = []
cut_physical_topics = []
for file in os.listdir():
    count = 0
    for paragraph in physical:
        if paragraph[0] == file:
            if count < minimum_paragraphs:
                cut_physical.append(paragraph)
                count += 1
    count = 0
    topic = []
    for paragraph in cut_physical:
        if paragraph[0] == file:
            count += 1
            topic.append(paragraph[1:])
    cut_physical_topics.append((file, topic))
    cut_physical_count.append((file, count))
#print(cut_physical_count)
#print(cut_physical_topics[1])

#Create dataset of mental and physical pages, then create labels for the dataset by mental and physical.
data = cut_mental_topics + cut_physical_topics
labels = [0]*len(cut_mental_topics) + [1]*len(cut_physical_topics)

#Create training dataset. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
##Shuffle data. https://stackoverflow.com/a/52776718
##Create test set.
sp = StratifiedShuffleSplit(n_splits=1, train_size=0.55, random_state=1)
for temp_index, test_index in sp.split(data, labels):
    data_temp_prel = [data[i] for i in temp_index]
    data_test_prel = [data[i] for i in test_index]
##Create validation set. https://stackoverflow.com/a/42932524
sp = StratifiedShuffleSplit(n_splits=1, test_size=0.1/0.55, random_state=1)
for train_index, val_index in sp.split(data_temp_prel, [labels[i] for i in temp_index]):
    data_train_prel = [data_temp_prel[i] for i in train_index]    
    data_val_prel = [data_temp_prel[i] for i in val_index]

data_train = []
for topic, content in data_train_prel:
    for label, paragraph in content:
        data_train.append((topic, label, paragraph))
data_val = []
for topic, content in data_val_prel:
    for label, paragraph in content:
        data_val.append((topic, label, paragraph))
data_test = []
for topic, content in data_test_prel:
    for label, paragraph in content:
        data_test.append((topic, label, paragraph))

print("Data training set:", len(data_train))
print("Data validation set:", len(data_val))
print("Data testing set:", len(data_test))

#Create labels and strings for the training set.
train_page = []
train_label = []
train_content = []
for page, label, content in data_train:
    train_page.append(page)
    train_label.append(label)
    train_content.append(content)
dump((train_page, train_label, train_content), '../../paragraph_train.joblib')

#Create labels and strings for the validation set.
val_page = []
val_label = []
val_content = []
for page, label, content in data_val:
    val_page.append(page)
    val_label.append(label)
    val_content.append(content)
dump((val_page, val_label, val_content), '../../paragraph_val.joblib')

#Create labels and strings for the testing set.
test_page = []
test_label = []
test_content = []
for page, label, content in data_test:
    test_page.append(page)
    test_label.append(label)
    test_content.append(content)
dump((test_page, test_label, test_content), '../../paragraph_test.joblib')

#Create training vector.
##https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer
vectorize = TfidfVectorizer(stop_words='english')
train_vector = vectorize.fit_transform(train_content)
print("Matrix of Training Vector: \n", train_vector.toarray())
print("Shape of Training Vector: \n", train_vector.shape)

#MODEL BUILDING
print("*Training Classifiers*")
#Make predictions using Naive Bayes Classifier.
nb = MultinomialNB().fit(train_vector, train_label)
nb_predicted = nb.predict(train_vector)
#print('Naive Bayes Predictions (Actual, Predicted):')
#i = 0
#for doc, category in zip(train_content, nb_predicted):
#    print('%r => %s' % (train_label[i], category))
#    i += 1
nb_acc = np.mean(nb_predicted == train_label)
print("Training Naive Bayes Accuracy:", nb_acc)

#Make predictions using a Support Vector Machine.
svm = SGDClassifier(loss='hinge', penalty='l2',
                    alpha=1e-3, random_state=42,
                    max_iter=5, tol=None)

svm.fit(train_vector, train_label)
svm_predicted = svm.predict(train_vector)
svm_acc = np.mean(svm_predicted == train_label)
print("Training SVM Accuracy:", svm_acc)

#Get top features of SVM. https://stackoverflow.com/a/34236002
feature_array = np.array(vectorize.get_feature_names_out())
train_tfidf_sorting = np.argsort(svm.coef_.flatten())
n = 10
train_top_n = feature_array[train_tfidf_sorting][:n]
train_bottom_n = feature_array[train_tfidf_sorting][-1*n:]
print("Mental Feature Weights:", train_tfidf_sorting[:n])
print("Mental Training Features: ", train_top_n)
print("Physical Feature Weights:", train_tfidf_sorting[-1*n:])
print("Physical Training Features: ", train_bottom_n)

#Make predictions using a Random Forest Classifier.
rf = RandomForestClassifier(max_depth=2, random_state=0)
rf.fit(train_vector, train_label)
rf_predicted = rf.predict(train_vector)
rf_acc = np.mean(rf_predicted == train_label)
print("Training Random Forest Accuracy:", rf_acc)

#Make predictions using a Logistic Regression.
lg = LogisticRegression(random_state=0).fit(train_vector, train_label)
lg_predicted = lg.predict(train_vector)
lg_acc = np.mean(lg_predicted == train_label)
print("Training Logistic Regression Accuracy:", lg_acc)

#Get top features of Logistic Regression. https://stackoverflow.com/a/34236002
feature_array = np.array(vectorize.get_feature_names_out())
train_tfidf_sorting = np.argsort(lg.coef_.flatten())
n = 10
train_top_n = feature_array[train_tfidf_sorting][:n]
train_bottom_n = feature_array[train_tfidf_sorting][-1*n:]
print("Mental Feature Weights:", train_tfidf_sorting[:n])
print("Mental Training Features: ", train_top_n)
print("Physical Feature Weights:", train_tfidf_sorting[-1*n:])
print("Physical Training Features: ", train_bottom_n)

#DATA ANALYSIS
#Validation
##Naive Bayes
val_vector = vectorize.transform(val_content)
val_nb_predicted = nb.predict(val_vector)
val_nb_acc = np.mean(val_nb_predicted == val_label)
print("Validation Naive Bayes Accuracy:", val_nb_acc)
##SVM
val_svm_predicted = svm.predict(val_vector)
val_svm_acc = np.mean(val_svm_predicted == val_label)
print("Validation SVM Accuracy:", val_svm_acc)
##Random Forest
val_rf_predicted = rf.predict(val_vector)
val_rf_acc = np.mean(val_rf_predicted == val_label)
print("Validation Random Forest Accuracy:", val_rf_acc)
##Logistic Regression
val_lg_predicted = lg.predict(val_vector)
val_lg_acc = np.mean(val_lg_predicted == val_label)
print("Validation Logistic Regression Accuracy:", val_lg_acc)

#Testing
##Naive_Bayes
test_vector = vectorize.transform(test_content)
test_nb_predicted = nb.predict(test_vector)
test_nb_acc = np.mean(test_nb_predicted == test_label)
print("Testing Naive Bayes Accuracy:", test_nb_acc)
##SVM
test_svm_predicted = svm.predict(test_vector)
test_svm_acc = np.mean(test_svm_predicted == test_label)
print("Testing SVM Accuracy:", test_svm_acc)
##Random Forest
test_rf_predicted = rf.predict(test_vector)
test_rf_acc = np.mean(test_rf_predicted == test_label)
print("Testing Random Forest Accuracy:", test_rf_acc)
##Logistic Regression
test_lg_predicted = lg.predict(test_vector)
test_lg_acc = np.mean(test_lg_predicted == test_label)
print("Testing Logistic Regression Accuracy:", test_lg_acc)

#Save models.
dump((vectorize, nb, svm, rf, lg), '../../paragraph_models.joblib')