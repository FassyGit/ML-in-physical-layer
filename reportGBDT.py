import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
traindata = pd.read_csv('stock_train_data_20171111.csv')
testdata = pd.read_csv('stock_test_data_20171111.csv')
X_train=traindata[:]
del X_train['label']
del X_train['id']
del X_train['era']
del X_train['weight']
y_train=traindata['label']
X_test=testdata[:]
del X_test['id']
weight=traindata['weight'].values
logistic_model = BaggingClassifier(GradientBoostingClassifier(random_state=10),
                                   max_samples=0.5,max_features=0.5)
logistic_model.fit(X_train, y_train,sample_weight=weight)
fitted_test= logistic_model.predict_proba(X_test)[:, 1]
save = pd.DataFrame({'id':testdata['id'],'proba':fitted_test})  
save.to_csv('resultGBDT.csv',index=False,sep=',') 