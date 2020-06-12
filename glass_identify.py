import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score,train_test_split,KFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

names=['Id Number','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','Class']

df = pd.read_csv('glass-data.csv',names=names)

#df.hist()

#df.plot(kind='density',subplots=True,layout=(4,4),sharex=False,sharey=False)

#corr = df.corr()
#sns.heatmap(corr,vmin=-1,vmax=1)

array = df.values
x = array[:,:-1]
y = array[:,-1]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=7)

models=[]

models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('CART',DecisionTreeClassifier()))
models.append(('KN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

results=[]
names=[]

for name,model in models:
    
    kfold = KFold(n_splits=10,random_state=7,shuffle=True)
    cv_result = cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy') 
    results.append(cv_result)
    names.append(name)
    
    msg = "%s: %f %f"%(name,cv_result.mean()*100,cv_result.std())
    print(msg)
    

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',
LogisticRegression(solver='liblinear'))])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA',
LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',
KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',
DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB',
GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM',
SVC(gamma='auto'))])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std())
    print(msg)
    
model=SVC(gamma='auto')
c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
param_grid = dict(C=c_values,kernel=kernel_values)
kfold = KFold(n_splits=10,random_state=7,shuffle=True)
grid  = GridSearchCV(estimator=model,param_grid=param_grid,cv=kfold,scoring='accuracy')
grid_result = grid.fit(x_train,y_train)
print(grid_result.best_score_ , grid_result.best_params_)

model = SVC(C=0.3,kernel='linear')
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions)*100)

model = DecisionTreeClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions)*100)

model = KNeighborsClassifier()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(accuracy_score(y_test,predictions)*100)
