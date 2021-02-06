import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
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
df=pd.read_csv('clean_news_articles.csv')

avg=0
for i in range(100):
# GET A TRAIN TEST SPLIT (set seed for consistent results)
    training_data, testing_data = train_test_split(df,test_size = 0.25,random_state = 0)


    # GET LABELS
    Y_train=training_data['label'].values
    Y_test=testing_data['label'].values

    # GET FEATURES: (X_train=train_feature_set) ,(X_test=test_feature_set),(feature_transformer=cv)
    X_train,X_test,feature_transformer=extract_features(df,'text_without_stopwords',training_data,testing_data,type="counts")


# _________________________________LOGISTIC REGRESSION CLASSIFIER__________________________________________

    scikit_log_reg = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=1000)
    model=scikit_log_reg.fit(X_train,Y_train) # learning with the features we fount and the train set

    ytest = np.array(Y_test)
    print("Result of Logistic Regression Model...")
    # confusion matrix and classification report(precision, recall, F1-score)
    print(classification_report(ytest, model.predict(X_test)))
    print(confusion_matrix(ytest, model.predict(X_test)))
    avg+=scikit_log_reg.score(X_train, Y_train)
print("average accuracy score:  " , avg/100)