import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from bayesclassifier import fit, predict

#load data
pathtogender = "/Users/leonievanstappen/Documents/school/master/CAI/categorization/genderdata.csv"
df = pd.read_csv(pathtogender, header=None, keep_default_na=False)
#prep data for count vectorizer
df.columns = ["text", "gender"]
df["gender"] = df["gender"].apply(lambda x: x.strip().upper()) #normalize labels: trailing spaces, lowercase letters, etc.
male = df.loc[df["gender"] == "M"]
female = df.loc[df["gender"] == "F"]
#matrix met counts
vect = CountVectorizer(analyzer='word', max_features=100)
docs = df['text']
X = (vect.fit_transform(docs)).toarray()
#vector met labels
le = LabelEncoder()
y = le.fit_transform(df['gender'])
#split data for train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=55)

print(fit(X_train, y_train))
print(predict(X_test))

#TEST
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
mnb.score(X_test, y_test)
