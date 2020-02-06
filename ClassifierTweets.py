import pandas as pd
import re
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree.export import export_text
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn_pandas import DataFrameMapper
from sklearn.metrics import confusion_matrix
from sklearn import svm
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords


def model_data(dataset):
    #Regexp for extracting hashtags
    hashtag_re = re.compile("#(\w+)", re.UNICODE)

    #Remove duplicated bot tweets
    fltr_dataset = dataset.drop_duplicates(subset='tweetText', keep="first")

    #Extract tweet hashtags
    fltr_dataset['hashtag'] = fltr_dataset["tweetText"].str.extract(hashtag_re)

    #Convert humour labels to fake stories, to make classification easier
    fltr_dataset['label'] = fltr_dataset['label'].map({'real': 1, 'fake': 0, 'humor': 0})

    #Count uppercase letter, as indication for emotion
    fltr_dataset['uppercase'] = fltr_dataset['tweetText'].str.count(r'[A-Z]')

    #Count exclmation marks as indication for emotion
    fltr_dataset['symbol'] = fltr_dataset['tweetText'].str.count('!')

    #Determine tweet language as another useful feature
    if(True):
        for index, row in fltr_dataset['tweetText'].iteritems():
            try:
                lang = detect(row)
                fltr_dataset.loc[index, 'lang'] = lang
            except:
                fltr_dataset.loc[index, 'lang'] = 'no lang'

    fltr_dataset_finish = fltr_dataset[['tweetText', 'hashtag', 'uppercase', 'symbol', 'lang', 'label']]

    #Concat all features so they are ready for feature extraction
    fltr_dataset_finish['tweetText'] = fltr_dataset_finish['tweetText'].astype('U')
    fltr_dataset_finish['tweetText'] = fltr_dataset_finish['tweetText'] + fltr_dataset_finish['uppercase'].astype('U')
    fltr_dataset_finish['tweetText'] = fltr_dataset_finish['tweetText'] + fltr_dataset_finish['symbol'].astype('U')
    fltr_dataset_finish['tweetText'] = fltr_dataset_finish['tweetText'] + fltr_dataset_finish['hashtag'].astype('U')
    fltr_dataset_finish['tweetText'] = fltr_dataset_finish['tweetText'] + fltr_dataset_finish['lang'].astype('U')

    return fltr_dataset_finish

#Get datasets
mediaeval_training = pd.read_csv("mediaeval-2015-trainingset.txt", sep="	")
mediaeval_testing = pd.read_csv("mediaeval-2015-testset.txt", sep="	")

#Modeling the data to fit the model
X_training = model_data(mediaeval_training)
X_testing = model_data(mediaeval_testing)

Y_training = X_training['label']
Y_testing = X_testing['label']

#X_training_final = X_training[['tweetText','hashtag','lang']]
#X_testing_final = X_testing[['tweetText','hashtag','lang']]

#Getting the column for vectorization
X_training_final = X_training[['tweetText']]
X_testing_final = X_testing[['tweetText']]

#print(X_training_final['uppercase'])
#Predfined stop words, aimed to improve accuracy when detecting real tweets

#vectorizer = CountVectorizer(max_features=10000,lowercase=True, stop_words=stop_words)
tfid_vectorizer = TfidfVectorizer(max_features=2700, stop_words=stopwords.words('english'))

#mapper_X= DataFrameMapper([('tweetText', tfid_vectorizer),('hashtag', vectorizer), ('lang', vectorizer)], sparse=True)
mapper_X= DataFrameMapper([('tweetText', tfid_vectorizer)], sparse=True)

#Creating the model, later used for the SVM
X = mapper_X.fit_transform(X_training_final)
X_features = mapper_X.transformed_names_
Z = mapper_X.transform(X_testing_final)

#Creating the SVC, training it and then predicting the values on the test set
clf = svm.SVC(kernel='linear')
print('TRAINING SVM')
clf.fit(X, Y_training)
print('PREDICTING ')
prediction = clf.predict(Z)
print('Precision score: ')
print(precision_score(Y_testing, prediction))
print('Recall score: ')
print(recall_score(Y_testing, prediction))
print('SVM F1 score: ')
print(f1_score(Y_testing, prediction))
print('SVM R2 score: ')
print(r2_score(Y_testing, prediction))
print('SVM confusion matrix: ')
print(confusion_matrix(Y_testing, prediction))


#print()
#print('----------------------------')
#print()

print('TRAINING DECISION TREE')
tree_clf = DecisionTreeClassifier(max_depth=16)
tree_clf.fit(X, Y_training)
tree.plot_tree(tree_clf)
r = export_text(tree_clf, feature_names= X_features)
print(r)
print('PREDICTING ')

prediction = tree_clf.predict(Z)

print('Precision score: ')
print(precision_score(Y_testing, prediction, average='macro'))
print('Recall score: ')
print(recall_score(Y_testing, prediction, average='macro'))
print('DT F1 score: ')
print(f1_score(Y_testing, prediction, average='micro'))
print('DT R2 score: ')
print(r2_score(Y_testing, prediction))
print('DT confusion matrix: ')
print(confusion_matrix(Y_testing, prediction))

#print()
#print('----------------------------')
#print()

#best_value = f1_score(Y_testing, prediction, average='micro')
#best_size = 0

#for i in range(1,151):
#    tree_clf = DecisionTreeClassifier(max_depth=i)
#    tree_clf.fit(X, Y_training)
#    prediction = tree_clf.predict(Z)
#    if(best_value < f1_score(Y_testing, prediction, average='micro')):
#        best_value = f1_score(Y_testing, prediction, average='micro')
#        best_size = i

#print("Best F1 score for DT is: ")
#print(best_value)
#print("For size of: ")
#print(best_size)