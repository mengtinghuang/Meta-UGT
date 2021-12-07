from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,matthews_corrcoef, confusion_matrix
import time
from sklearn.model_selection import train_test_split
from itertools import product
import pickle
from sklearn.externals import joblib
#svm_grid
parameters_svm = {'kernel': ['rbf'], 'gamma': np.logspace(-15,3,10,base=2),'C':np.logspace(-5,9,8,base=2),'class_weight':['balanced'] }    
#nn_grid    
parameters_nn = {"learning_rate":["constant"],   #learning_rate:{"constant","invscaling","adaptive"} default constant
                  "max_iter":[10000],
                  "hidden_layer_sizes":[(50,),(100,),(200,),(300,),(500,)],
                  "alpha":10.0 ** -np.arange(1, 7),
                  "activation":["relu"],  #"identity","tanh","relu","logistic"
                  "solver":["adam"]}           #"lbfgs","adam""sgd"
#knn_grid
parameters_knn = {"n_neighbors":range(3,10,2),"weights":['distance',"uniform"]}
#rf_grid
parameters_rf = {"n_estimators":range(10,121,10),
                   "criterion" : ["gini","entropy"],
                   "oob_score": ["True","False"],
                   "class_weight":["balanced_subsample","balanced"]}

np.random.seed(4)

#dt_grid
parameters_dt={"criterion":['entropy','gini'],"splitter":['best','random'],"max_depth":range(5,50,5),}
#lr_grid
parameters_lr={"solver":["liblinear","lbfgs","newton-cg","sag"],"C":np.arange(0.01,10.01,0.5)}
#et_grid
parameters_et={"n_estimators":range(10,101,10),"max_depth":range(5,50,5), "criterion" : ["gini","entropy"]}
model_map = {"svm":SVC, "knn":KNeighborsClassifier,"nn":MLPClassifier,"rf":RandomForestClassifier,"dt":DecisionTreeClassifier,"lr":LogisticRegression,"et":ExtraTreesClassifier}    
fp_map = {"maccs":"MACCSFP", "fp": "FP","subfp":"SubFP","pubchemfp":"PubchemFP"}
parameter_grid = {"svm":parameters_svm, "knn":parameters_knn,"nn":parameters_nn,"rf":parameters_rf,"dt":parameters_dt,"lr":parameters_lr,"et":parameters_et}
cv = StratifiedKFold(n_splits = 10,shuffle=True,random_state = 1)

#evaluation metrics
def accuracy(y_true,y_pred):
    return accuracy_score(y_true,y_pred)

def precision(y_true,y_pred):
    return precision_score(y_true,y_pred,pos_label=1,average="binary")

def recall(y_true,y_pred):
    return recall_score(y_true,y_pred,pos_label=1,average="binary")    
    
def auc(y_true,y_scores):
    return roc_auc_score(y_true,y_scores)        

def mcc(y_true,y_pred):
    return matthews_corrcoef(y_true, y_pred) 
    
def new_confusion_matrix(y_true,y_pred):
    return confusion_matrix(y_true,y_pred,labels = [0,1]) 

def sp(y_true,y_pred):
    cm = new_confusion_matrix(y_true,y_pred)
    return cm[0,0] * 1.0 / (cm[0,0] + cm[0,1])

def classifer_generator(tuned_parameters,method,train_x,train_y,n_jobs = 10):
    """
    Return the best model and the parameters of the model'
    """
    if method == SVC:
        grid = GridSearchCV(method(probability=True,random_state = 1), param_grid=tuned_parameters, scoring = "accuracy", cv = cv, n_jobs = n_jobs)
    elif method == KNeighborsClassifier:
        grid = GridSearchCV(method(), param_grid=tuned_parameters, scoring = "accuracy", cv = cv, n_jobs = n_jobs)
    else:
        grid = GridSearchCV(method(random_state = 1), param_grid=tuned_parameters, scoring = "accuracy", cv = cv, n_jobs = n_jobs)
    grid.fit(train_x,train_y)
    return grid.best_estimator_ , grid.best_params_    

def data_reader(filename):
    'Read fingerprint file'
    data = pd.read_csv(filename,header=None).values
    X = data[:,1:]
    y = data[:,0]
    return X, y


def cv_results(best_model,train_x,train_y):
    'Return the performance of the cross-validation'
    y_true = []
    y_pred = []
    y_scores = []
    for train_index,test_index in cv.split(train_x,train_y):
        x_train,x_test = train_x[train_index],train_x[test_index]
        y_train,y_test = train_y[train_index],train_y[test_index]
        m = best_model.fit(x_train,y_train)
        y_true.extend(y_test)
        y_pred.extend(m.predict(x_test))
        y_scores.extend(m.predict_proba(x_test)[:,1])
    return accuracy(y_true,y_pred),precision(y_true,y_pred),recall(y_true,y_pred),auc(y_true,y_scores),mcc(y_true,y_pred),sp(y_true,y_pred),y_true,y_pred

def test_results(best_model,test_x,test_y):
    'Return the performance of the test validation'
    y_true = test_y
    y_pred = best_model.predict(test_x)
    y_scores =  best_model.predict_proba(test_x)[:,1]
    return accuracy(y_true,y_pred),    precision(y_true,y_pred),recall(y_true,y_pred),    auc(y_true,y_scores),mcc(y_true,y_pred),sp(y_true,y_pred),y_true,y_pred

def search_best_model(training_data,method_name,test_data):
    'Return {"model": best_model, "cv": metrics_of_cv,"tv": metrics_of_test,"parameter": best_parameter", "method": method_name}'
    train_x = training_data[0]
    train_y = training_data[1]
    tuned_parameters = parameter_grid[method_name]
    method = model_map[method_name]
    cg =  classifer_generator(tuned_parameters,method,train_x,train_y)
    best_model = cg[0]
    cv_metrics = cv_results(best_model,train_x,train_y)[:6]
    result = {'model': best_model, 'cv': cv_metrics, 'method': method_name}
    if test_data:
        test_x = test_data[0]
        test_y = test_data[1]
        test_metrics = test_results(best_model,test_x,test_y)[:6]
        result['tv'] = {'model': best_model, 'cv': test_metrics, 'method': method_name}
    return result,result['tv'],cg[1],best_model


def main(model_list,fp_list,targets,bits):
    model_names = product(model_list,fp_list,targets,bits)
    metrics_file = open("cv_metrics.txt","w")
    test_file = open("test_metrics.txt","w")
    parameter_file = open("parameter.txt","w")
    metrics_file.write("Target\tFingerprint\tMethod\tbit\tAccuracy\tPrecision\tRecall\tAUC\tMCC\tSP\n")
    test_file.write("Target\tFingerprint\tMethod\tbit\tAccuracy\tPrecision\tRecall\tAUC\tMCC\tSP\n")
    parameter_file.write("Method\tParameter\n")
    for method_name,fp_name,target,bit in model_names:
        print(method_name,fp_name,bit)
        training_data = data_reader(target+'_train_'+fp_name+bit+'.csv')
        test_data = data_reader(target+'_test_'+fp_name+bit+'.csv')
        model_results = search_best_model(training_data,method_name,test_data)[0]
        test_result = search_best_model(training_data,method_name,test_data)[1]
        best_model = search_best_model(training_data,method_name,test_data)[3]
        pickle_file = open (method_name+'_'+fp_name+'_'+bit+'.pkl','wb')
        s =pickle.dump(best_model, pickle_file)
        cv_res = [str(x) for x in model_results['cv']]
        cv_test =[str(x) for x in test_result['cv']]
        metrics_file.write('%s\t%s\t%s\t%s\t%s\n'%(target,fp_name,method_name,bit,'\t'.join(cv_res)))
        test_file.write('%s\t%s\t%s\t%s\t%s\n'%(target,fp_name,method_name,bit,'\t'.join(cv_test)))
        parameter_file.write('%s\t%s\t%s\t%s\t%s\n'%(target,fp_name,method_name,bit,search_best_model(training_data,method_name,test_data)[2]))
        print('best_parameters:' + str(search_best_model(training_data,method_name,test_data)[2]))
    metrics_file.close()

if __name__ == '__main__':
    start = time.clock()
    model_list =["svm","nn","lr","et"]
    fp_list = ["RDKFingerprint", "TopoTorsion", "Morgan", "AtomPairs"]
    bits = ["512","1024","2048"]
    targets = ['ugt']
    main(model_list,fp_list,targets,bits)
    print("time consumption:",time.clock()-start,"The end!")	