import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_top_k_predictions(model,X_test,k):

    # get probabilities instead of predicted labels, since we want to collect top 3
    probs = model.predict_proba(X_test)

    # GET TOP K PREDICTIONS BY PROB - note these are just index
    best_n = np.argsort(probs, axis=1)[:,-k:]

    # GET CATEGORY OF PREDICTIONS
    preds=[[model.classes_[predicted_cat] for predicted_cat in prediction] for prediction in best_n]

    # REVERSE CATEGORIES - DESCENDING ORDER OF IMPORTANCE
    preds=[ item[::-1] for item in preds]

    return preds

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


# GET A TRAIN TEST SPLIT (set seed for consistent results)
training_data, testing_data = train_test_split(df,random_state = 2000)


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
# ________________________________END LOGISTIC REGRESSION CLASSIFIER__________________________________________


print("___________________________________________________________________________________________")


# _______________________________________________SVM CLASSIFIER__________________________________________
svclassifier = SVC(kernel='linear',max_iter=1000)
svclassifier.fit(X_train, Y_train)
y_pred = svclassifier.predict(X_test)
ytest = np.array(Y_test)
print("Result of SVM Model...")
print(classification_report(ytest,y_pred))
print(confusion_matrix(ytest,y_pred))
# _______________________________________________END SVM CLASSIFIER______________________________________________


print("___________________________________________________________________________________________")


# _______________________________________________RANDOM FOREST CLASSIFIER__________________________________________
clf=RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)

# prediction on test set
y_pred=clf.predict(X_test)
ytest = np.array(Y_test)

print("Result of Random Forest Model...")
print(classification_report(ytest,y_pred))
print(confusion_matrix(ytest,y_pred))
# _______________________________________________END RANDOM FOREST CLASSIFIER__________________________________________


print("___________________________________________________________________________________________")


# _______________________________________________KNN CLASSIFIER__________________________________________
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
ytest = np.array(Y_test)
print("Result of KNN Model...")
print(classification_report(ytest, y_pred))
print(confusion_matrix(ytest, y_pred))
# _______________________________________________END KNN CLASSIFIER__________________________________________


print("___________________________________________________________________________________________")


# _______________________________________________ADABOOST CLASSIFIER__________________________________________
classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),n_estimators=200)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
ytest = np.array(Y_test)
print("Result of ADABOOST Model...")
print(classification_report(ytest, y_pred))
print(confusion_matrix(ytest, y_pred))
# _______________________________________________END ADABOOST CLASSIFIER__________________________________________






#
# # Visualising the Training set results
# X_set, y_set = X_train, Y_train
# X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                      np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
# plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#              alpha = 0.75, cmap = ListedColormap(('blue','red')))
# plt.xlim(X1.min(), X1.max())
# plt.ylim(X2.min(), X2.max())
# for i, j in enumerate(np.unique(y_set)):
#     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                 c = ListedColormap(('blue','red'))(i), label = j)
# plt.title('Logistic Regression (Training set)')
# plt.xlabel('Fake')
# plt.ylabel('Real')
# plt.legend()
# plt.show()


# # GET TOP K PREDICTIONS
# preds=get_top_k_predictions(model,X_test,10)
# print()
# #
# GET PREDICTED VALUES AND GROUND TRUTH INTO A LIST OF LISTS
# eval_items=collect_preds(Y_test,preds)
#
# # GET EVALUATION NUMBERS ON TEST SET -- HOW DID WE DO?
# print("Starting evaluation...")
# accuracy=compute_accuracy(eval_items)
# mrr_at_k=compute_mrr_at_k(eval_items)
