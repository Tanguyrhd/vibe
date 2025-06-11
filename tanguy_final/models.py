# pip install -U imbalanced-learn --q
# pip install lightgbm --q
# pip install xgboost --q



from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.combine import SMOTETomek
from lightgbm import LGBMClassifier




def vect(df, max_features=20000, ngram_range=(1, 2), stop_words=list(ENGLISH_STOP_WORDS), col='clean_text'):
    vectorizer = TfidfVectorizer(max_features=max_features,
                             ngram_range=ngram_range,
                             stop_words=stop_words
                             )

    X = vectorizer.fit_transform(df[col])
    return X



def xbg_smote(df, col, X):
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    smote_tomek = SMOTETomek()
    X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    return y_pred, y_test



def xgb(df, col, X):
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test



def lgbm_smote(df, col, X):
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = LGBMclassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred, y_test



def lgbm(df, col, X):
    y = df[col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    smote_tomek = SMOTETomek()
    X_res, y_res = smote_tomek.fit_resample(X_train, y_train)
    model = LGBMclassifier()
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    return y_pred, y_test
