from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from lightgbm import LGBMClassifier

# 훈련 / 테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2, stratify=y)

ss = StandardScaler()
X_train_ss = ss.fit_transform(X_train)
X_test_ss = ss.transform(X_test)

pca = PCA()
X_train_pca = pca.fit_transform(X_train_ss, y_train)
X_test_pca = pca.transform(X_test_ss)

lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X_train_ss, y_train)
X_test_lda = lda.transform(X_test, y_test)


def modeling(X_train, y, model, feature_tran='pca'):
    
    # 검증용 세트 분할
    #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size = 0.2, stratify=y)

    # 평가 점수를 담을 데이터프레임 생성
    scores = pd.DataFrame()

    # 모델 생성
    g_model = globals()[model]()
   
    
    # 모델 감지
    if model == 'KNeighborsClassifier':

        param_grid = {'n_neighbors':[3,5,7,9],
                      'algorithm':['auto', 'ball_tree','kd_tree','brute'],
                      'metric':['euclidean','manhattan','minkowski']}        
        scores['model'] = ['Knn']
        
    elif model == 'SVC':

        param_grid = {'C': [0.01, 0.1, 1.0, 10.0],
                       'kernel': ['linear', 'rbf'],
                       'gamma': ['scale', 'auto']}        
        scores['model'] = ['SVC']
            
    elif model == 'LogisticRegression':

        param_grid = {'penalty':['l1','l2','elasticnet',None],
                      'C':[0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                      'solver':['lbfgs','liblinear']}        
        scores['model'] = ['Logistic classifier']
        
    elif model == 'RandomForestClassifier':

        param_grid = {'n_estimators': [50, 100, 200],
                      'max_depth': [None, 10, 20],
                      'max_features': ['auto', 'sqrt'],
                      'bootstrap': [True, False]}        
        scores['model'] = ['Random forest']
        
    elif model == 'GradientBoostingClassifier':
        
        param_grid = {'loss': ['log_loss', 'exponential'],
                     'learning_rate': [0.01, 0.1, 1.0, 10],
                     'criterion': ['friedman_mse', 'squared_error']}
        
        scores['model'] = ['Gradient Boost Classifier']
        
    elif model == 'HistGradientBoostingClassifier':

        param_grid = {'max_iter': [100, 200, 300],
                      'max_depth': [3, 5],
                      'learning_rate': [0.01, 0.1, 1.0, 10.0]}        
        scores['model'] = ['Hist Random forest']
        
    elif model == 'XGBClassifier':

        param_grid = {'n_estimators': [50, 100, 200],
                      'max_depth': [3, 4, 5],
                      'learning_rate': [0.01, 0.1, 10.0]}        
        scores['model'] = ['XG Boost']
        
    elif model == 'LGBMClassifier':
        
        param_grid = {'n_estimators':[100, 200, 500],
                     'learning_rate': [0.01, 0.1, 1.0, 10],
                     'max_depth':[1,2,3]}
        scores['model'] = ['Light GBM']
        
    else:
        print('잘못된 입력입니다.')
        return  
        
    # 그리드 서치 모델링
    gs = GridSearchCV(estimator=g_model, param_grid=param_grid,
                      scoring='accuracy', cv=5, refit=True, n_jobs=-1)
    
    gs = gs.fit(X_train, y_train)

    best = gs.best_estimator_
    best.fit(X_train, y_train)
    pred_train = best.predict(X_train)
    pred_test = best.predict(X_test)

    scores['Best_parameters'] = [gs.best_params_]

    scores['Accuracy_Train'] = accuracy_score(y_train, pred_train)
    scores['Accuracy_Test'] = accuracy_score(y_test, pred_test)

    scores['Recall_Train'] = recall_score(y_train, pred_train)
    scores['Recall_Test'] = recall_score(y_test, pred_test)

    scores['F1_Train'] = f1_score(y_train, pred_train)
    scores['F1_Test'] = f1_score(y_test, pred_test)
    
    
    ConfusionMatrixDisplay.from_estimator(best, X, y)
    plt.title( scores['model'].values[0])
    plt.show()
    return scores
