class machine_learning:
    
    def __init__(self, x_train, y_train, x_test, y_test, type='classification'):
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.type = type
        self.models = {'lr': None,
                       #'K-Nearest Neighbors': None,
                       'svm': None,
                       'gb': None,
                       'rf': None,
                      'dt': None}
    
    
    def classification(self,
                       knn_n_neighbors=5, knn_weights='uniform', knn_algorithm='auto', knn_leaf_size=30, knn_p=2, knn_metric='minkowski', knn_metric_params=None, knn_n_jobs=None,
                       lr_penalty='l2', lr_dual=False, lr_tol=1e-4, lr_C=1.0, lr_fit_intercept=True, lr_intercept_scaling=1, lr_class_weight=None, lr_random_state=None, lr_solver='lbfgs', lr_max_iter=100, lr_multi_class='auto', lr_verbose=0, lr_warm_start=False, lr_n_jobs=None, lr_l1_ratio=None,
                       dt_criterion='gini', dt_splitter='best', dt_max_depth=None, dt_min_samples_split=2, dt_min_samples_leaf=1, dt_min_weight_fraction_leaf=0.0, dt_max_features=None, dt_random_state=None, dt_max_leaf_nodes=None, dt_min_impurity_decrease=0.0, dt_class_weight=None, dt_ccp_alpha=0.0,
                       svm_C=1.0, svm_kernel='rbf', svm_degree=3, svm_gamma='scale', svm_coef0=0.0, svm_shrinking=True, svm_probability=True, svm_tol=1e-3, svm_cache_size=200, svm_class_weight=None, svm_verbose=False, svm_max_iter=-1, svm_decision_function_shape='ovr', svm_break_ties=False, svm_random_state=None,
                       gb_loss='deviance', gb_learning_rate=0.1, gb_n_estimators=100, gb_subsample=1.0, gb_criterion='friedman_mse', gb_min_samples_split=2, gb_min_samples_leaf=1, gb_min_weight_fraction_leaf=0.0, gb_max_depth=3, gb_min_impurity_decrease=0.0, gb_init=None, gb_random_state=None, gb_max_features=None, gb_verbose=0, gb_max_leaf_nodes=None, gb_warm_start=False, gb_validation_fraction=0.1, gb_n_iter_no_change=None, gb_tol=1e-4, gb_ccp_alpha=0.0,
                       rf_n_estimators=100, rf_criterion='gini', rf_max_depth=None, rf_min_samples_split=2, rf_min_samples_leaf=1, rf_min_weight_fraction_leaf=0.0, rf_max_features='auto', rf_max_leaf_nodes=None, rf_min_impurity_decrease=0.0, rf_bootstrap=True, rf_oob_score=False, rf_n_jobs=None, rf_random_state=None, rf_verbose=0, rf_warm_start=False, rf_class_weight=None, rf_ccp_alpha=0.0):
        
        # 각 모델의 인자와 값을 딕셔너리로 정리
        knn_params = {'n_neighbors': knn_n_neighbors,
                      'weights': knn_weights,
                      'algorithm': knn_algorithm,
                      'leaf_size': knn_leaf_size,
                      'p': knn_p,
                      'metric': knn_metric,
                      'metric_params': knn_metric_params,
                      'n_jobs': knn_n_jobs}
        
        lr_params = {'penalty': lr_penalty,
                     'dual': lr_dual,
                     'tol': lr_tol,
                     'C': lr_C,
                     'fit_intercept': lr_fit_intercept,
                     'intercept_scaling': lr_intercept_scaling,
                     'class_weight': lr_class_weight,
                     'random_state': lr_random_state,
                     'solver': lr_solver,
                     'max_iter': lr_max_iter,
                     'multi_class': lr_multi_class,
                     'verbose': lr_verbose,
                     'warm_start': lr_warm_start,
                     'n_jobs': lr_n_jobs,
                     'l1_ratio': lr_l1_ratio}
        
        dt_params = {'criterion': dt_criterion,
                     'splitter': dt_splitter,
                     'max_depth': dt_max_depth,
                     'min_samples_split': dt_min_samples_split,
                     'min_samples_leaf': dt_min_samples_leaf,
                     'min_weight_fraction_leaf': dt_min_weight_fraction_leaf,
                     'max_features': dt_max_features,
                     'random_state': dt_random_state,
                     'max_leaf_nodes': dt_max_leaf_nodes,
                     'min_impurity_decrease': dt_min_impurity_decrease,
                     #'min_impurity_split': dt_min_impurity_split,
                     'class_weight': dt_class_weight,
                     #'presort': dt_presort,
                     'ccp_alpha': dt_ccp_alpha}
        
        svm_params = {'C': svm_C,
                      'kernel': svm_kernel,
                      'degree': svm_degree,
                      'gamma': svm_gamma,
                      'coef0': svm_coef0,
                      'shrinking': svm_shrinking,
                      'probability': svm_probability,
                      'tol': svm_tol,
                      'cache_size': svm_cache_size,
                      'class_weight': svm_class_weight,
                      'verbose': svm_verbose,
                      'max_iter': svm_max_iter,
                      'decision_function_shape': svm_decision_function_shape,
                      'break_ties': svm_break_ties,
                      'random_state': svm_random_state}
        
        gb_params = {'loss': gb_loss,
                     'learning_rate': gb_learning_rate,
                     'n_estimators': gb_n_estimators,
                     'subsample': gb_subsample,
                     'min_samples_split': gb_min_samples_split,
                     'min_samples_leaf': gb_min_samples_leaf,
                     'min_weight_fraction_leaf': gb_min_weight_fraction_leaf,
                     'max_depth': gb_max_depth,
                     'min_impurity_decrease': gb_min_impurity_decrease,
                     #'min_impurity_split': gb_min_impurity_split,
                     'init': gb_init,
                     'random_state': gb_random_state,
                     'max_features': gb_max_features,
                     'verbose': gb_verbose,
                     'max_leaf_nodes': gb_max_leaf_nodes,
                     'warm_start': gb_warm_start,
                     #'presort': gb_presort,
                     'validation_fraction': gb_validation_fraction,
                     'n_iter_no_change': gb_n_iter_no_change,
                     'tol': gb_tol,
                     'ccp_alpha': gb_ccp_alpha}
        
        rf_params = {'n_estimators': rf_n_estimators,
                     'criterion': rf_criterion,
                     'max_depth': rf_max_depth,
                     'min_samples_split': rf_min_samples_split,
                     'min_samples_leaf': rf_min_samples_leaf,
                     'min_weight_fraction_leaf': rf_min_weight_fraction_leaf,
                     'max_features': rf_max_features,
                     'max_leaf_nodes': rf_max_leaf_nodes,
                     'min_impurity_decrease': rf_min_impurity_decrease,
                     #'min_impurity_split': rf_min_impurity_split,
                     'bootstrap': rf_bootstrap,
                     'oob_score': rf_oob_score,
                     'n_jobs': rf_n_jobs,
                     'random_state': rf_random_state,
                     'verbose': rf_verbose,
                     'warm_start': rf_warm_start,
                     'class_weight': rf_class_weight,
                     'ccp_alpha': rf_ccp_alpha}
        
        if self.type == 'classification':
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.svm import SVC
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.ensemble import RandomForestClassifier
            
            # 모델 생성 및 훈련
            knn = KNeighborsClassifier(**knn_params)
            lr = LogisticRegression(**lr_params)
            svm = SVC(**svm_params)
            gb = GradientBoostingClassifier(**gb_params)
            rf = RandomForestClassifier(**rf_params)
            dt = DecisionTreeClassifier(**dt_params)
            
            
            self.models['lr'] = lr.fit(self.x_train, self.y_train)
            #self.models['K-Nearest Neighbors'] = knn.fit(self.x_train, self.y_train)
            self.models['svm'] = svm.fit(self.x_train, self.y_train)
            self.models['gb'] = gb.fit(self.x_train, self.y_train)
            self.models['rf'] = rf.fit(self.x_train, self.y_train)
            self.models['dt'] = dt.fit(self.x_train, self.y_train)
        
        return knn, lr, svm, gb, rf, dt

    
    def classify_eval(self):
        for model_name, model in self.models.items():
            pred = model.predict(self.x_test)
            pred_proba = model.predict_proba(self.x_test)

            confusion = confusion_matrix(self.y_test, pred)
            accuracy = accuracy_score(self.y_test, pred)
            precision = precision_score(self.y_test, pred)
            recall = recall_score(self.y_test, pred)
            f1 = f1_score(self.y_test, pred)
            roc_score = roc_auc_score(self.y_test, pred_proba[:, 1])

            print(f'{model_name}의 스코어s')
            print('오차행렬')
            print(confusion)
            print('정확도: {:.4f}'.format(accuracy))
            print('정밀도: {:.4f}'.format(precision))
            print('재현율: {:.4f}'.format(recall))
            print('F1스코어: {:.4f}'.format(f1))
            print('ROC AUC 값: {:.4f}'.format(roc_score))

            pred_proba_class1 = pred_proba[:, 1]
            precisions, recalls, thresholds = precision_recall_curve(self.y_test, pred_proba_class1)

            pre_recal = pd.DataFrame()
            threshold_boundary = thresholds.shape[0]
            pre_recal['임계값'] = np.round(thresholds, 2)
            pre_recal['정밀도'] = np.round(precisions[:threshold_boundary], 3)
            pre_recal['재현율'] = np.round(recalls[:threshold_boundary], 3)

            print('\n --- 정확도 / 정밀도 트레이드 오프 그래프 ---')
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, precisions[:-1], linestyle='--', label='precision')
            plt.plot(thresholds, recalls[:-1], label='recall')
            plt.xlabel('Threshold value')
            plt.ylabel('Precision and Recall value')
            plt.legend()
            plt.grid()
            plt.show()
            print(pre_recal.T)        
        
    def grid_search(self, params=None, model=None, n=None):
        
        from sklearn.model_selection import GridSearchCV
        
        # 모델 이름을 통해 모델 인스턴스를 가져옴
        model_instance = None
        for m in self.models:
            if m.__class__.__name__ == model:
                model_instance = m
                break
        
        if model_instance is None:
            raise ValueError(f'Model with name "{model}" not found in the models list.')
        
        
        grid_model = GridSearchCV(model_instance, param_grid=params, cv=n, refit=True)
        grid_model.fit(self.x_train, self.y_train)
        
        print('GridSearchCV 최적 파라미터: ', grid_model.best_params_)
        print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_model.best_score_))
    
        score_df = pd.DataFrame(grid_model.cv_results_)
        estimator_df = pd.DataFrame(grid_model.best_estimator_)
    
        #  grid_model 평가
        self.classify_eval(x_test, y_test, grid_model)
    
        return estimator_df, score_df
        
    def model_attributes(self, model):
        
         # 모델 이름을 통해 모델 인스턴스를 가져옴
        model_instance = None
        for m in self.models:
            if m.__class__.__name__ == model:
                model_instance = m
                break
        
        if model_instance is None:
            raise ValueError(f'Model with name "{model}" not found in the models list.')

        attributes = [attr for attr in dir(model_instance) if not callable(getattr(model_instance, attr)) and not attr.startswith("__")]
        
        for attr in attributes:
            print(f'{attr} : ', getattr(model_instance.attr)) 
        
        
