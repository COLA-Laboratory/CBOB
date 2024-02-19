# models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import neural_network

# datasets
from sklearn.datasets import load_diabetes, load_digits, load_breast_cancer
from sklearn.datasets import fetch_covtype, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

import numpy as np
import xgboost as xgb
import pickle
import sys

from .data_observation import SimEnvironmentMultiConstraints


class SimEnvironmentSklearn(SimEnvironmentMultiConstraints):
    def __init__(self,
                 cons_partial=True,
                 classification=False,
                 infeasible_value=None,
                 feasible_value=None,
                 virtual_value=2.0,
                 dataset='house',
                 method='rf',
                 memory_ratio=0.5):
        self._method = method
        self._dataset = dataset
        self._memory_threshold = self.compute_threshold(memory_ratio)
        super().__init__(constraint_num=1, classification=classification, cons_partial=cons_partial,
                         virtual_value=virtual_value, infeasible_value=infeasible_value, feasible_value=feasible_value)

    def observer(self, query_point):
        # define the objective
        memory_test = Memory(self._dataset)

        fwd_r = np.zeros(len(query_point[:, 0]))
        ctrl_r = np.zeros(len(query_point[:, 0]))
        for i in range(len(query_point[:, 0])):
            if self._method == 'rf':
                fwd_r[i], ctrl_r[i] = memory_test.rf_evaluation(query_point[i, :])
            elif self._method == 'xgb':
                fwd_r[i], ctrl_r[i] = memory_test.xgboost_evaluation(query_point[i, :])
            elif self._method == 'nn':
                fwd_r[i], ctrl_r[i] = memory_test.nn_evaluation(query_point[i, :])

        # define the constraints
        feasible_query_point = query_point[ctrl_r <= self._memory_threshold]
        obj = fwd_r[ctrl_r <= self._memory_threshold][:, None]
        y = ctrl_r - self._memory_threshold
        noise_var = ctrl_r * 0.0 + 1e-6
        for i in range(len(y)):
            if y[i] > 0:
                y[i] = np.log(1 + y[i])
            else:
                y[i] = - np.log(- y[i] + 1)
        y = y.reshape(-1, 1)
        noise_var = noise_var.reshape(-1, 1)
        cons = np.hstack((y, noise_var))

        print("searching obj = ", obj)
        print("searching cons = ", cons)

        constraints = {}
        constraints = {**constraints, **{self.CONSTRAINT[0]: (query_point, cons)}}
        return {**{self.OBJECTIVE: (feasible_query_point, obj)}, **constraints}

    def compute_threshold(self, memory_ratio):
        threshold = None
        if self._method == 'rf':

            # digit: quantile0.3:  14965.0 	quantile0.5:  38548.5 	quantile0.7:  108875.69999999998
            if self._dataset == 'digit':
                if memory_ratio == 0.3:
                    threshold = 15000
                elif memory_ratio == 0.5:
                    threshold = 40000
                elif memory_ratio == 0.7:
                    threshold = 110000
                else:
                    raise ValueError("The memory threshold can only set by {0.3, 0.5, 0.7}!")

            elif self._dataset  == 'cancer':
                if memory_ratio == 0.3:
                    threshold = 5000
                elif memory_ratio == 0.5:
                    threshold = 12000
                elif memory_ratio == 0.7:
                    threshold = 28000
                else:
                    raise ValueError("The memory threshold can only set by {0.3, 0.5, 0.7}!")

            # cover: quantile0.3:  26935.999999999985        quantile0.5:  186908.0  quantile0.7:  2586217.899999999
            elif self._dataset == 'cover':
                if memory_ratio == 0.3:
                    threshold = 27000
                elif memory_ratio == 0.5:
                    threshold = 187000
                elif memory_ratio == 0.7:
                    threshold = 2586000
                else:
                    raise ValueError("The memory threshold can only set by {0.3, 0.5, 0.7}!")

            # house: quantile0.3:  16867.8   quantile0.5:  68451.0   quantile0.7:  340545.89999999985
            elif self._dataset == 'house':
                if memory_ratio == 0.3:
                    threshold = 17000
                elif memory_ratio == 0.5:
                    threshold = 68000
                elif memory_ratio == 0.7:
                    threshold = 340000
                else:
                    raise ValueError("The memory threshold can only set by {0.3, 0.5, 0.7}!")
        elif self._method == 'xgb':
            # quantile0.3:  39749.29999999998         quantile0.5:  123094.0  quantile0.7:  356713.39999999985
            if self._dataset == 'digit':
                if memory_ratio == 0.3:
                    threshold = 40000
                elif memory_ratio == 0.5:
                    threshold = 123000
                elif memory_ratio == 0.7:
                    threshold = 356000

            # quantile0.3:  15514.4   quantile0.5:  49073.0   quantile0.7:  150878.7
            elif self._dataset == 'house':
                if memory_ratio == 0.3:
                    threshold = 15500
                elif memory_ratio == 0.5:
                    threshold = 49000
                elif memory_ratio == 0.7:
                    threshold = 150000
                else:
                    raise ValueError("The memory threshold can only set by {0.3, 0.5, 0.7}!")

            elif self._dataset == 'diabetes':
                if memory_ratio == 0.3:
                    threshold = 8400
                elif memory_ratio == 0.5:
                    threshold = 18700
                elif memory_ratio == 0.7:
                    threshold = 52000

            elif self._dataset == 'steel':
                if memory_ratio == 0.3:
                    threshold = 9000
                elif memory_ratio == 0.5:
                    threshold = 18300
                elif memory_ratio == 0.7:
                    threshold = 47400
                else:
                    raise ValueError("The memory threshold can only set by {0.3, 0.5, 0.7}!")
        elif self._method == 'nn':
            # digit: quantile0.3:  59236.6   quantile0.5:  107265.5  quantile0.7:  218061.2
            if self._dataset == 'digit':
                if memory_ratio == 0.3:
                    threshold = 59236.6
                elif memory_ratio == 0.5:
                    threshold = 107265.5
                elif memory_ratio == 0.7:
                    threshold = 218061.2
                else:
                    raise ValueError("The memory threshold can only set by {0.3, 0.5, 0.7}!")
            # quantile0.3:  23551.0   quantile0.5:  46885.5   quantile0.7:  94148.99999999999
            elif self._dataset == 'diabetes':
                if memory_ratio == 0.3:
                    threshold = 23551.0
                elif memory_ratio == 0.5:
                    threshold = 46885.5
                elif memory_ratio == 0.7:
                    threshold = 94149.0
                else:
                    raise ValueError("The memory threshold can only set by {0.3, 0.5, 0.7}!")
        return threshold


class Memory:
    def __init__(self, dataset='cancer'):

        X_train = None
        y_train = None
        X_test = None
        y_test = None
        self.classification = True
        if dataset == 'digit':
            X, y = load_digits(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'cancer':
            X, y = load_breast_cancer(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'diabetes':
            self.classification = False
            X, y = load_diabetes(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'cover':
            X, y = fetch_covtype(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'house':
            self.classification = False
            X, y = fetch_california_housing(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        elif dataset == 'steel':
            self.classification = False
            X, y = fetch_california_housing(return_X_y=True)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=1
            )
        else:
            print("invalid dataset.")
            exit()

        self.dataset = {
            "trainX": X_train,
            "trainY": y_train,
            "testX": X_test,
            "testY": y_test,
        }

    def rf_evaluation(self, hyperparameters):
        # max_depth: 1-50 int, log
        # min_samples_split 2 - 128 int, log
        # n_estimators 0-64 int, log
        # min_samples_leaf 0-20 int
        # max_features (0,1) float
        max_depth = np.floor(np.power(10, hyperparameters[0])).astype(int)
        min_samples_split = np.floor(np.power(2, hyperparameters[1])).astype(int)
        n_estimators = np.floor(np.power(10, hyperparameters[2])).astype(int)
        min_samples_leaf = np.floor(hyperparameters[3]).astype(int)
        max_features = hyperparameters[4]
        if self.classification:
            classifier = RandomForestClassifier(max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                max_features=max_features,
                                                min_samples_leaf=min_samples_leaf,
                                                # criterion=criterion,
                                                n_estimators=n_estimators,
                                                )

            classifier.fit(self.dataset['trainX'], self.dataset['trainY'])
            p = pickle.dumps(classifier)
            memory_cost = sys.getsizeof(p)
            pred = classifier.predict(self.dataset['testX'])

            accuracy = accuracy_score(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return -accuracy, memory_cost
        else:
            regressor = RandomForestRegressor(max_depth=max_depth,
                                                min_samples_split=min_samples_split,
                                                max_features=max_features,
                                                min_samples_leaf=min_samples_leaf,
                                                # criterion=criterion,
                                                n_estimators=n_estimators,
                                                )

            regressor.fit(self.dataset['trainX'], self.dataset['trainY'])
            p = pickle.dumps(regressor)
            memory_cost = sys.getsizeof(p)
            pred = regressor.predict(self.dataset['testX'])

            accuracy = mean_squared_error(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return accuracy, memory_cost


    def xgboost_evaluation(self, hyperparameters):
        # eta: (2^**-10)-1. log, float
        # max_depth: 1-15   int
        # colsample_bytree 0.01-1. float
        # lambda_reg (2^**-10)-2**10 log, float
        # alpha_reg (2^**-10)-2**10 log, float
        # min_child_weight 1-2**7 log, float
        # n_estimator 1-2^**8 log, int
        eta = np.power(2, hyperparameters[0])
        max_depth = np.floor(hyperparameters[1]).astype(int)
        colsample_bytree = hyperparameters[2]
        lambda_reg = np.power(2, hyperparameters[3])
        alpha_reg = np.power(2, hyperparameters[4])
        min_child_weight = np.power(2, hyperparameters[5])
        n_estimator = np.floor(np.power(2, hyperparameters[6])).astype(int)

        if self.classification:
            classifier = xgb.XGBClassifier(
                learning_rate=eta,
                n_estimators=n_estimator,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                colsample_bytree=colsample_bytree,
                reg_lambda=lambda_reg,
                reg_alpha=alpha_reg,
                random_state=1,
                n_jobs=1,
            )
            classifier.fit(self.dataset['trainX'], self.dataset['trainY'])
            pred = classifier.predict(self.dataset['testX'])
            p = pickle.dumps(classifier)
            memory_cost = sys.getsizeof(p)

            accuracy = accuracy_score(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return -accuracy, memory_cost
        else:
            regressor = xgb.XGBRegressor(
                learning_rate=eta,
                n_estimators=n_estimator,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                colsample_bytree=colsample_bytree,
                reg_lambda=lambda_reg,
                reg_alpha=alpha_reg,
                random_state=1,
                n_jobs=1,
            )
            regressor.fit(self.dataset['trainX'], self.dataset['trainY'])
            pred = regressor.predict(self.dataset['testX'])
            p = pickle.dumps(regressor)
            memory_cost = sys.getsizeof(p)

            accuracy = mean_squared_error(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return accuracy, memory_cost

    def nn_evaluation(self, hyperparameters):
        # hidden_layer_sizes (a, b): (2^2)-(2^8). log, int
        # batch_size 2^2-2^8 log, int
        # alpha (10^-8)-(10^-3) log, float
        # learning_rate_init (10^-5)-(10^0), log, float
        # tol (10^-6)-(10^-2) log, float
        # beta_1 0.-0.9999 float
        # beta_2 0.-0.9999 float
        hidden_layer_sizes_a = np.round(np.power(2, hyperparameters[0])).astype(int)
        hidden_layer_sizes_b = np.round(np.power(2, hyperparameters[1])).astype(int)
        batch_size = np.round(np.power(2, hyperparameters[2])).astype(int)
        alpha = np.power(10, hyperparameters[3])
        learning_rate_init = np.power(10, hyperparameters[4])
        tol = np.power(10, hyperparameters[5])
        beta_1 = hyperparameters[6]
        beta_2 = hyperparameters[7]
        if self.classification:
            classifier = neural_network.MLPClassifier(
                hidden_layer_sizes=(hidden_layer_sizes_a, hidden_layer_sizes_b),
                batch_size=batch_size,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                tol=tol,
                beta_1=beta_1,
                beta_2=beta_2,
                random_state=1,
            )
            classifier.fit(self.dataset['trainX'], self.dataset['trainY'])
            p = pickle.dumps(classifier)
            pred = classifier.predict(self.dataset['testX'])

            memory_cost = sys.getsizeof(p)
            accuracy = accuracy_score(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return -accuracy, memory_cost
        else:
            regressor = neural_network.MLPRegressor(
                hidden_layer_sizes=(hidden_layer_sizes_a, hidden_layer_sizes_b),
                batch_size=batch_size,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                tol=tol,
                beta_1=beta_1,
                beta_2=beta_2,
                random_state=1,
            )
            regressor.fit(self.dataset['trainX'], self.dataset['trainY'])
            pred = regressor.predict(self.dataset['testX'])
            p = pickle.dumps(regressor)
            memory_cost = sys.getsizeof(p)

            accuracy = mean_squared_error(pred, self.dataset['testY'])
            print("accuracy: ", accuracy, "     model size: ", memory_cost)
            return accuracy, memory_cost





