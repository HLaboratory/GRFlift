import sys
sys.path.append("../../../CausalModel")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from TreeModel.model.grf import UpliftTreeRegressor

class UpliftRandomForestClassifier:
    """ Uplift Random Forest for Classification Task.

    Parameters
    ----------
    n_estimators : integer, optional (default=10)
        The number of trees in the uplift random forest.

    evaluationFunction : string
        Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS'.

    max_features: int, optional (default=10)
        The number of features to consider when looking for the best split.

    random_state: int, optional (default=2019)
        The seed used by the random number generator.

    max_depth: int, optional (default=5)
        The maximum depth of the tree.

    min_samples_leaf: int, optional (default=100)
        The minimum number of samples required to be split at a leaf node.

    min_samples_treatment: int, optional (default=10)
        The minimum number of samples required of the experiment group to be split at a leaf node.

    n_reg: int, optional (default=10)
        The regularization parameter defined in Rzepakowski et al. 2012, the
        weight (in terms of sample size) of the parent node influence on the
        child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.

    control_name: string
        The name of the control group (other experiment groups will be regarded as treatment groups)

    normalization: boolean, optional (default=True)
        The normalization factor defined in Rzepakowski et al. 2012,
        correcting for tests with large number of splits and imbalanced
        treatment and control splits

    Outputs
    ----------
    df_res: pandas dataframe
        A user-level results dataframe containing the estimated individual treatment effect.
    """

    # https://causalml.readthedocs.io/en/latest/methodology.html#uplift-tree
    def __init__(self,
                 n_estimators=20,
                 datas_mode = 'random',
                 max_features=0.5,
                 max_datas = 1.0,
                 random_state=2020,
                 max_depth=5,
                 min_samples_leaf=100,
                 min_samples_treatment=10,
                 n_reg=10,
                 evaluationFunction='KL',
                 control_name='control',
                 normalization=True,
                 is_constrained=True,
                 show_log=False,
                 x_names=None,
                 n_jobs=None):
        """
        Initialize the UpliftRandomForestClassifier class.
        """
        self.classes_ = {}
        self.n_estimators = n_estimators
        self.datas_mode = datas_mode
        self.max_features = max_features
        self.max_datas = max_datas
        self.random_state = random_state
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.evaluationFunction = evaluationFunction
        self.control_name = control_name
        self.is_constrained = is_constrained
        self.show_log = show_log
        self.n_jobs = n_jobs
        self.aucc_score = None
        self.qini_score = None
        self.x_name = x_names
        # Create forest
        # 创建n个随机森林分类器
        self.uplift_forest = []

        for _ in range(n_estimators):
            uplift_tree = UpliftTreeRegressor(
                max_features=self.max_features, max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_treatment=self.min_samples_treatment,
                n_reg=self.n_reg,
                evaluationFunction=self.evaluationFunction,
                control_name=self.control_name,
                normalization=normalization,
                x_names=self.x_name,
                if_prune=None)
            self.uplift_forest.append(uplift_tree)

    def fit(self,  X, treatment, y, c):
        """
        Fit the UpliftRandomForestClassifier.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.

        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.
        """
        # 随机种子
        np.random.seed(self.random_state)
        # Get treatment group keys
        # 获得所有的treatment组的名字
        treatment_group_keys = list(set(treatment))
        treatment_group_keys.remove(self.control_name)
        treatment_group_keys.sort()
        self.classes_ = {}
        for i, treatment_group_key in enumerate(treatment_group_keys):
            self.classes_[treatment_group_key] = i
            
        # random or average
        if self.datas_mode == 'random':
            # 随机森林多棵树并行计算
            if not self.n_jobs or self.n_jobs <= 1:
                for tree_i in range(len(self.uplift_forest)):
                    print(f'第{tree_i}颗树')
                    bt_index = np.random.choice(len(X), int(len(X)*self.max_datas), replace=False)
                    x_train_bt = X[bt_index]
                    y_train_bt = y[bt_index]
                    c_train_bt = c[bt_index]
                    treatment_train_bt = treatment[bt_index]
                    # x_train_bt = X
                    # y_train_bt = y
                    # c_train_bt = c
                    # treatment_train_bt = treatment
                    self.uplift_forest[tree_i].fitted_uplift_tree = self.uplift_forest[tree_i].fit(x_train_bt,
                                                                                                treatment_train_bt,
                                                                                                y_train_bt,
                                                                                                c_train_bt)
            else:
                from joblib import Parallel, delayed, parallel_backend
                tasks = []
                if self.n_jobs >= 32:
                    self.n_jobs = 32
                for tree_i in range(len(self.uplift_forest)):
                    
                    bt_index = np.random.choice(len(X), int(len(X)*self.max_datas), replace=False)
                    x_train_bt = X[bt_index]
                    y_train_bt = y[bt_index]
                    c_train_bt = c[bt_index]
                    treatment_train_bt = treatment[bt_index]
                    tree = self.uplift_forest[tree_i]
                    tasks.append(delayed(self.multiple_thread_rf)(tree_i, tree, x_train_bt,
                                                                treatment_train_bt, y_train_bt, c_train_bt))
                with parallel_backend("loky", n_jobs=self.n_jobs):
                    for result in Parallel(prefer="processes", n_jobs=self.n_jobs, pre_dispatch='1 * n_jobs')(tasks):
                        self.uplift_forest[result[0]].fitted_uplift_tree = result[1]
                        
        if self.datas_mode == 'average':
            piece_num = int(len(X)/self.n_estimators)
            # 随机森林多棵树并行计算
            if not self.n_jobs or self.n_jobs <= 1:
                
                for tree_i in range(len(self.uplift_forest)):
                    print(f'第{tree_i}颗树')
                    
                    x_train_bt = X[tree_i*piece_num: (tree_i+1)*piece_num]
                    y_train_bt = y[tree_i*piece_num: (tree_i+1)*piece_num]
                    c_train_bt = c[tree_i*piece_num: (tree_i+1)*piece_num]
                    treatment_train_bt = treatment[tree_i*piece_num: (tree_i+1)*piece_num]
                    self.uplift_forest[tree_i].fitted_uplift_tree = self.uplift_forest[tree_i].fit(x_train_bt,
                                                                                                treatment_train_bt,
                                                                                                y_train_bt,
                                                                                                c_train_bt)
            else:
                from joblib import Parallel, delayed, parallel_backend
                tasks = []
                if self.n_jobs >= 32:
                    self.n_jobs = 32
                for tree_i in range(len(self.uplift_forest)):
                    
                    x_train_bt = X[tree_i*piece_num: (tree_i+1)*piece_num]
                    y_train_bt = y[tree_i*piece_num: (tree_i+1)*piece_num]
                    c_train_bt = c[tree_i*piece_num: (tree_i+1)*piece_num]
                    treatment_train_bt = treatment[tree_i*piece_num: (tree_i+1)*piece_num]
                    tree = self.uplift_forest[tree_i]
                    tasks.append(delayed(self.multiple_thread_rf)(tree_i, tree, x_train_bt,
                                                                treatment_train_bt, y_train_bt, c_train_bt))
                with parallel_backend("loky", n_jobs=self.n_jobs):
                    for result in Parallel(prefer="processes", n_jobs=self.n_jobs, pre_dispatch='1 * n_jobs')(tasks):
                        self.uplift_forest[result[0]].fitted_uplift_tree = result[1]

    def multiple_thread_rf(self, tree_i=None, tree=None, X=None, treatment=None, y=None, c=None):
        return tree_i, tree.fit(X, treatment, y, c)

    def predict(self, X, n_jobs=None,full_output=False ):
        '''
        Returns the recommended treatment group and predicted optimal
        probability conditional on using the recommended treatment group.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        full_output : bool, optional (default=False)
            Whether the UpliftTree algorithm returns upliftScores, pred_nodes
            alongside the recommended treatment group and p_hat in the treatment group.

        Returns
        -------
        df_res : DataFrame, shape = [num_samples, (num_treatments + 1)]
            A DataFrame containing the predicted delta in each treatment group,
            the best treatment group and the maximum delta.
            :param is_generate_json:
            :param n_jobs:

        '''
        # 数据格式转换

        p_hat_optimal = []
        treatment_optimal = []
        pred_nodes = {}
        upliftScores = []

        df_res = pd.DataFrame()
        y_pred_ensemble = dict()
        y_pred_list = np.zeros((X.shape[0], len(self.classes_)))
        tree_list = []

        for xi in range(len(X)):
            for tree_i in range(len(self.uplift_forest)):
                if tree_i == 0:
                    pred_leaf, upliftScore = self.uplift_forest[tree_i].classify(X[xi],
                                                                                 self.uplift_forest[tree_i].fitted_uplift_tree,
                                                                                 dataMissing=False)
                else:
                    tmp_pred_leaf, tmp_uplift_score = self.uplift_forest[tree_i].classify(X[xi],
                                                                                          self.uplift_forest[tree_i].fitted_uplift_tree,
                                                                                          dataMissing=False)
                    for k, v in tmp_pred_leaf.items(): # {treat1: score, treat2: score}
                        pred_leaf[k] += v # {treat1: score求和, treat2: score求和}
                    for v in range(len(tmp_uplift_score)):
                        upliftScore[v] += v
            # avg
            avg_pred_leaf = {k: v/len(upliftScore)for k, v in pred_leaf.items()} # {treat1: score均值, treat2: score均值}
            avg_upliftScore = [i/len(upliftScore) for i in upliftScore]
            opt_treat = max(avg_pred_leaf, key=avg_pred_leaf.get)
            p_hat_optimal.append(avg_pred_leaf[opt_treat])
            treatment_optimal.append(opt_treat)

            if full_output:
                if xi == 0:
                    for key_i in avg_pred_leaf:
                        pred_nodes[key_i] = [avg_pred_leaf[key_i]]
                else:
                    for key_i in avg_pred_leaf:
                        pred_nodes[key_i].append(avg_pred_leaf[key_i])
                upliftScores.append(avg_upliftScore)

        if full_output:
            return treatment_optimal, p_hat_optimal, upliftScores, pred_nodes
        else:
            return treatment_optimal, p_hat_optimal


    def multiple_thread_predict(self, tree_i=None, X=None):
        _, _, _, y_pred_full = self.uplift_forest[tree_i].predict(X=X, full_output=True)
        return tree_i, y_pred_full


if __name__ == '__main__':
    # urf = UpliftRandomForestClassifier(n_estimators=2,
    #                                    max_features=0.8,
    #                                    max_depth=3,
    #                                    evaluationFunction='gmv', n_jobs=2)
    # X = np.random.rand(1000, 10)
    # y = np.random.random(1000, 1)
    # cost = np.zeros(shape=(1000, 1))
    # treatment = np.array([['treatment']*500 + ['control']*500])
    # UpliftRandomForestClassifier.fit(X, treatment, y, cost)

    uplift_model = UpliftRandomForestClassifier(
        n_estimators=5,
        datas_mode = 'random',
        max_features=1.0,
        max_datas = 0.2,
        max_depth=5,
        min_samples_leaf=100,
        min_samples_treatment=10,
        n_reg=100,
        evaluationFunction='Gmv',
        control_name='control',
        x_names=np.random.rand(1000, 10),
        n_jobs=1
    )

    uplift_model.fit(X=np.random.rand(1000, 10),
                        treatment=np.array([['treatment']*500 + ['control']*500]),
                        y=np.random.random(1000, 1),
                        c=np.zeros(shape=(1000, 1)))
