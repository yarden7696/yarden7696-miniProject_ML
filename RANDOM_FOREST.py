import random

import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def extract_features(df,field,training_data,testing_data,type="binary"):
    """Extract features using different methods"""

    print("Extracting features and creating vocabulary...")

    if "binary" in type:

        # BINARY FEATURE REPRESENTATION
        cv= CountVectorizer(binary=True, max_df=0.95)
        cv.fit_transform(training_data[field].values)

        train_feature_set=cv.transform(training_data[field].values)
        test_feature_set=cv.transform(testing_data[field].values)

        return train_feature_set,test_feature_set,cv

    elif "counts" in type:

        # COUNT BASED FEATURE REPRESENTATION
        cv= CountVectorizer(binary=False) # no need df.max_df caz we do not have stopwords
        cv.fit_transform(training_data[field].values) # learning about the text_without_stopwords

        train_feature_set=cv.transform(training_data[field].values)
        test_feature_set=cv.transform(testing_data[field].values)
        return train_feature_set,test_feature_set,cv

    else:

        # TF-IDF BASED FEATURE REPRESENTATION
        tfidf_vectorizer=TfidfVectorizer(use_idf=True, max_df=0.95)
        tfidf_vectorizer.fit_transform(training_data[field].values)

        train_feature_set=tfidf_vectorizer.transform(training_data[field].values)
        test_feature_set=tfidf_vectorizer.transform(testing_data[field].values)

        return train_feature_set,test_feature_set,tfidf_vectorizer

#read dataset

df = pd.read_csv('clean_news_articles.csv')
avg=0
n=100
for i in range(n):
    #df.sample(frac=1)
    df = sklearn.utils.shuffle(df)

    # GET A TRAIN TEST SPLIT (set seed for consistent results)
    training_data, testing_data = train_test_split(df,test_size = 0.25,random_state = 42,shuffle=True)

    df['text_without_stopwords'] = str(df['hasImage'])+df['text_without_stopwords']


    # GET LABELS
    Y_train=training_data['label'].values
    Y_test=testing_data['label'].values

    # GET FEATURES: (X_train=train_feature_set) ,(X_test=test_feature_set),(feature_transformer=cv)
    X_train,X_test,feature_transformer=extract_features(df,'text_without_stopwords',training_data,testing_data,type="counts")


# _________________________________RANDOM FOREST CLASSIFIER__________________________________________

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, Y_train)

    # prediction on test set
    y_pred = clf.predict(X_test)
    ytest = np.array(Y_test)

    print("Result of Random Forest Model...")
    print(classification_report(ytest, y_pred))
    cm=(confusion_matrix(ytest, y_pred))

    print(cm)
    avg+=accuracy_score(Y_test, y_pred)
print("average accuracy score:  " , avg/n)
df.groupby('label').text_without_stopwords.count().plot.bar(ylim=0)

plt.show()



fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Fake', 'Predicted Real'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Fake', 'Actual Real'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.title('Random Forest average score=%s '% (avg/n) )
plt.show()
