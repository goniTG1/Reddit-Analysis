#### Machine Learning part


import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import neighbors
from sklearn import svm
import pandas as pd


"""# PREDICTING POPULARITY OF A POST
likes, comments, upvote ratio

linear models
"""

# FACTORES QUE PUEDEN INTERVENIR EN LA POPULARIDAD DE UN POST:

# 1- EPOCA DEL AÑO
# 2- NÚMERO DE POSTS ANTERIORES DEL USUARIO
# 3- TEMA DEL POST
# 4- LONGITUD DEL TÍTULO
# 5- CONTIENE ALGUN TIPO DE MEDIA
# 6- CONTIENE SOLO MEDIA
# 7- LA MEDIA QUE CONTIENE ES UN REDDIT VIDEO

def use_model(df, model_selected=None, error_metric=['mse']):

    prediction_df = df.drop(['post id', 'author', 'name', 'year', 'day', 'media only', 'distinguised',
                             'ups', 'Topic', 'clicked', 'New_Title', 'Comments', 'Eng Title',
                             'Eng Comments', 'Verified user'], axis=1)

    def int_or_none(x):
        if x == '':
            return 0
        else:
            return float(x)

    prediction_df['gilded'] = prediction_df['gilded'].apply(int_or_none)
    prediction_df['downs'] = prediction_df['downs'].apply(int_or_none)
    prediction_df['total_awards_received'] = prediction_df['total_awards_received'].apply(int_or_none)
    prediction_df['pinned'] = prediction_df['pinned'].apply(int_or_none)
    prediction_df['num_crossposts'] = prediction_df['num_crossposts'].apply(int_or_none)
    prediction_df["media"] = prediction_df['media'].apply(lambda x: isinstance(x, str))
    prediction_df["selftext"] = prediction_df['selftext'].apply(lambda x: len(x) > 0)
    prediction_df["title_length"] = prediction_df["title"].apply(lambda x: len(x.split()))
    prediction_df = prediction_df.drop("title", axis=1)

    dummy = pd.get_dummies(prediction_df["media"])
    dummy = dummy.rename(columns={True: "Media"})
    dummy2 = pd.get_dummies(prediction_df["selftext"])
    dummy2 = dummy2.rename(columns={True: "Selftext"})

    prediction_df = prediction_df.drop(["media", "selftext"], 1)
    prediction_df = pd.concat([prediction_df, dummy["Media"], dummy2["Selftext"]], axis=1)

    # we only keep the topics that appear more than once
    # number_of_topics = prediction_df["Topic"].value_counts().apply(lambda x: x>2).value_counts()[1]
    train = prediction_df.sample(frac=0.7, random_state=200).drop(['score'], axis=1)
    test = prediction_df.drop(train.index).drop(['score'], axis=1)
    train_target = prediction_df.sample(frac=0.7, random_state=200)["score"]
    test_target = prediction_df.drop(train.index)["score"]

    if model_selected == 'linear':
        regr = linear_model.LinearRegression()
        regr.fit(train, train_target)
        pred = regr.predict(test)

    if model_selected == 'Random Forest':
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        rf.fit(train, train_target)
        pred = list(rf.predict(test))

    if model_selected == 'knn':
        n_neighbors=5
        knn=neighbors.KNeighborsRegressor(n_neighbors,weights='uniform')
        knn.fit(train,train_target)
        pred=list(knn.predict(test))


    if model_selected == 'bayesian':
        reg = linear_model.BayesianRidge()
        reg.fit(train,train_target)
        pred=list(reg.predict(test))

    if model_selected == 'decision trees':
        dec = tree.DecisionTreeRegressor(max_depth=1)
        dec.fit(train,train_target)
        pred=list(dec.predict(test))

    if model_selected == 'svm':
        svm_reg=svm.SVR()
        svm_reg.fit(train,train_target)
        pred=list(svm_reg.predict(test))

    elif model_selected == 'comparison of all these models':
        regr = linear_model.LinearRegression()
        regr.fit(train, train_target)
        pred_linear= regr.predict(test)

        rf = RandomForestRegressor(n_estimators=1000, random_state=42)
        rf.fit(train, train_target)
        pred_rf = list(rf.predict(test))

        n_neighbors = 5
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')
        knn.fit(train, train_target)
        pred_knn = list(knn.predict(test))

        reg = linear_model.BayesianRidge()
        reg.fit(train, train_target)
        pred_bayes = list(reg.predict(test))

        dec = tree.DecisionTreeRegressor(max_depth=1)
        dec.fit(train, train_target)
        pred_decision = list(dec.predict(test))

        svm_reg = svm.SVR()
        svm_reg.fit(train, train_target)
        pred_svm = list(svm_reg.predict(test))

        return [pred_linear, pred_rf, pred_knn, pred_bayes, pred_decision, pred_svm], test_target

    final_error = []
    for error in error_metric:
        if error == 'mse':
            error_value = mean_squared_error(test_target, pred)
        if error == 'mae':
            error_value = mean_absolute_error(test_target, pred)
        if error == 'R-squared':
            error_value = r2_score(test_target, pred)
        final_error.append([error_value, error])

    return final_error



