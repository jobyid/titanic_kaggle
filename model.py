from sklearn.ensemble import RandomForestClassifier
import data_prep as dp
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

X_train, X_test, y_train, y_test = dp.X_train, dp.X_test, dp.y_train, dp.y_test
rf = RandomForestClassifier()
ada = AdaBoostClassifier()
gb = GradientBoostingClassifier()
kn = KNeighborsClassifier()
mlp = MLPClassifier(max_iter=15000)
gub = GaussianNB()


def random_forrest_raw():
    global rf
    rf.fit(X_train, y_train)
    s = rf.score(X_test,y_test)
    print("Random forrest score: {:.2f}%".format(s * 100))#, "\nwith params: ")#, rf.get_params())

def ada_raw():
    ada.fit(X_train,y_train)
    print("ADA boost score: {:.2f}%".format(ada.score(X_test, y_test) * 100))#, "\nwith params: ", ada.get_params())

def grad_raw():
    gb.fit(X_train,y_train)
    print("Gradient boost score: {:.2f}%".format(gb.score(X_test, y_test) * 100))#, "\nwith params: ", gb.get_params())

def kn_rwa():
    kn.fit(X_train,y_train)
    print("KNN score: {:.2f}%".format(gb.score(X_test, y_test) * 100))#, "\nwith params: ", kn.get_params())

def mlp_raw(test=X_test):
    mlp.fit(X_train,y_train)
    s = mlp.score(X_test,y_test)
    print("MlP score: {:.2f}%".format(s * 100), "\nwith params: ", mlp.get_params())
    pred = mlp.predict(test)
    return pred

def gaus_raw():
    gub.fit(X_train,y_train)
    s = gub.score(X_test, y_test)
    print("Gup score: {:.2f}%".format(s * 100))#, "\nwith params: ", gub.get_params())
# Classifying
#random_forrest_raw()

# grid search random forrest
def grid_search_random_forest():
    rf = RandomForestClassifier()
    params = {'criterion':('gini', 'entropy')}
    gs = GridSearchCV(rf,params)
    gs.fit(X_train,y_train)
    print("Best score accivedby gs: {:.2f}%".format(gs.best_score_ *100))
    print("Identified best params are: ",gs.best_params_)

def grid_search_gradient_boost():
    params = {'loss':('deviance', 'exponential'), 'criterion':('friedman_mse', 'mse', 'mae')}
    gsg = GridSearchCV(gb,params)
    gsg.fit(X_train,y_train)
    print("Best score achieved by gsg: {:.2f}%".format(gsg.best_score_ * 100))
    print("Identified best params are: ", gsg.best_params_)

def grid_nueral():
    params = {'activation':('identity', 'logistic', 'tanh', 'relu'),'solver':('lbfgs', 'sgd', 'adam'),'learning_rate':('constant', 'invscaling', 'adaptive')}
    gsn = GridSearchCV(mlp, params)
    gsn.fit(X_train,y_train)
    print("Best score achieved by gsn: {:.2f}%".format(gsn.best_score_ * 100))
    print("Identified best params are: ", gsn.best_params_)

#grid_nueral()

#grid_search_gradient_boost()

def make_sub_mission():
    f = dp.drop_columns(dp.tdf,test=True)
    X = dp.mean_age(f)
    X = dp.sc.fit_transform(X)
    #mlp.fit(X_train,y_train)
    pred_sub = mlp.predict(X)
    pre_df = pd.DataFrame(pred_sub, columns=["Survived"])
    print("pred", pre_df.describe())
    pre_df['PassengerId'] = dp.tdf.PassengerId
    print(pre_df.describe())
    pre_df.to_csv('submit.csv',index=False)


#random_forrest_raw()
#ada_raw()
#grad_raw()
#kn_rwa()
mlp_raw()
gaus_raw()
make_sub_mission()
