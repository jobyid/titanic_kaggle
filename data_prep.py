import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

df = pd.read_csv('train.csv')
tdf = pd.read_csv('test.csv')

df['Pclass']= df['Pclass'].replace([1,2],[25,8])
tdf['Pclass']= df['Pclass'].replace([1,2],[25,8])

def drop_columns(df=df, test=False):
       x_df = df[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
              'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
       # Name seems like something we can drop
       # cabin also seems droppable
       # ticket can be dropped
       # embarked can be droped
       # drop id
       # 'Parch',
       fx_df = df[[ 'Pclass', 'Sex', 'Age', 'SibSp',
               'Fare','Name']]
       fx_df.loc[(fx_df.Name.str.contains('Lady') ), 'Age'] = 8
       fx_df.loc[(fx_df.Name.str.contains('Lord')), 'Age'] = 8
       fx_df.loc[(fx_df.Name.str.contains('The Countess')), 'Age'] = 8
       # age has some missing values
       # Sex is categorical
       fx_df.Sex = fx_df.Sex.replace('male', 0).replace('female',5)
       if test:
              fx_df.Fare.fillna(value=df.Age.mean(), inplace=True)
       return fx_df

def group_age(df):
       df.loc[(df.Age < 18),'Age'] = 0
       df.loc[(df.Age >= 18), 'Age'] = 1
       return df

def mean_age(df = drop_columns()):
       df.Age.fillna(value=df.Age.mean(), inplace=True)
       group_age(df)
       X = df.iloc[:,:].values
       return X


def median_age(df = drop_columns()):
       df.Age.fillna(value=df.Age.median(), inplace=True)
       #add_weight_to_class(df)
       X = df.iloc[:, :].values
       return X


def frame_after_pca():
       pdf = df[[ 'Pclass', 'Sex']]
       pdf.Sex = pdf.Sex.replace('male', 0).replace('female', 1)
       X = pdf.iloc[:, :].values
       return X

X = mean_age()
y = np.array(df['Survived'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# grid se

def do_pca(X_, X_train_s, y_train_s):
       pca = PCA()
       pca._fit(X_)
       r = pca.explained_variance_ratio_
       print(r)
       x=['Pclass', 'Sex', 'Age', 'SibSp',
              'Parch', 'Fare']
       print("PCA values: ", pca.explained_variance_ratio_)
       sns.barplot(x =x ,y=r)
       plt.show()
       X_train, y_train = pca.fit_transform(X_train_s,y_train_s)
       return X_train, y_train

