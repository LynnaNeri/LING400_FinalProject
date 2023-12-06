import os
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
os.chdir('wiki_pages/mental_pages')
for file in os.listdir(): #https://www.geeksforgeeks.org/how-to-iterate-over-files-in-directory-using-python/
    if not file.startswith('.'):
        page = file
        cur_file = os.getcwd() + '/' + file
        contents = Path(cur_file).read_text() #https://stackoverflow.com/questions/3758147/easiest-way-to-read-write-a-files-content-in-python
        mental.append((page, 0, contents))

#Collect pages related to physical health.
physical = []
os.chdir('../physical_pages')
for file in os.listdir():
    if not file.startswith('.'):
        page = file
        cur_file = os.getcwd() + '/' + file
        contents = Path(cur_file).read_text()
        physical.append((page, 1, contents))

#Create dataset of mental and physical pages, then create labels for the dataset by mental and physical.
data = mental + physical
labels = [0]*len(mental) + [1]*len(physical)

#Ensure all datasets were completed.
print('Mental Health Pages:', len(mental))
print('Physical Health Pages:', len(physical))
print('Total Pages: ', len(data))
print('Total Labels:', len(labels))

#Create training dataset. https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
##Shuffle data. https://stackoverflow.com/a/52776718
##Create test set.
sp = StratifiedShuffleSplit(n_splits=1, train_size=0.55, random_state=1)
for temp_index, test_index in sp.split(data, labels):
    data_temp = [data[i] for i in temp_index]
    data_test = [data[i] for i in test_index]
##Create validation set. https://stackoverflow.com/a/42932524
sp = StratifiedShuffleSplit(n_splits=1, test_size=0.1/0.55, random_state=1)
for train_index, val_index in sp.split(data_temp, [labels[i] for i in temp_index]):
    data_train = [data_temp[i] for i in train_index]    
    data_val = [data_temp[i] for i in val_index]
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
dump((train_page, train_label, train_content), '../../page_train.joblib')

#Create labels and strings for the validation set.
val_page = []
val_label = []
val_content = []
for page, label, content in data_val:
    val_page.append(page)
    val_label.append(label)
    val_content.append(content)
dump((val_page, val_label, val_content), '../../page_val.joblib')

#Create labels and strings for the testing set.
test_page = []
test_label = []
test_content = []
for page, label, content in data_test:
    test_page.append(page)
    test_label.append(label)
    test_content.append(content)
dump((test_page, test_label, test_content), '../../page_test.joblib')

#Create training vector.
##https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html#sklearn.feature_extraction.text.TfidfVectorizer
vectorize = TfidfVectorizer(stop_words='english')
train_vector = vectorize.fit_transform(train_content)
print("Matrix of Training Vector: \n", train_vector.toarray())
print("Shape of Training Vector: \n", train_vector.shape)

#MODEL BUILDING
print("*Training Classifiers*")
#Make predictions using Naive Bayes Classifier.
nb = MultinomialNB().fit(train_vector, train_label) #https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
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
                    max_iter=5, tol=None) #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

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
rf = RandomForestClassifier(max_depth=2, random_state=0) #https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
rf.fit(train_vector, train_label)
rf_predicted = rf.predict(train_vector)
rf_acc = np.mean(rf_predicted == train_label)
print("Training Random Forest Accuracy:", rf_acc)

#Make predictions using a Logistic Regression.
lg = LogisticRegression(random_state=0).fit(train_vector, train_label) #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
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
dump((vectorize, nb, svm, rf, lg), '../../page_models.joblib')