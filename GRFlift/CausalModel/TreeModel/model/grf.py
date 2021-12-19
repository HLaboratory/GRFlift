import sys
sys.path.append("../../../CausalModel")

from tqdm import tqdm
from CausalModel.TreeModel.model.tree.baseTree import UpliftTree
from CausalModel.TreeModel.model.tree.splitRule import *
import numpy as np

# Uplift Tree Classifier
class UpliftTreeRegressor(object):
    """ Uplift Tree Classifier for Classification Task.

    A uplift tree classifier estimates the individual treatment effect by modifying the loss function in the
    classification trees.

    The uplift tree classifer is used in uplift random forest to construct the trees in the forest.

    Parameters
    ----------

    evaluationFunction : string
        Choose from one of the models: "DDP", "KL", "ED", "Chi", "CTS", "Roi", "Net"

    max_features: int, optional (default=10)
        The number of features to consider when looking for the best split.

    max_depth: int, optional (default=5)
        The maximum depth of the tree.

    min_samples_leaf: int, optional (default=100)
        The minimum number of samples required to be split at a leaf node.

    min_samples_treatment: int, optional (default=10)
        The minimum number of samples required of the experiment group to be split at a leaf node.

    n_reg: int, optional (default=10)
        The regularization parameter defined in Rzepakowski et al. 2012, the weight (in terms of sample size) of the
        parent node influence on the child node, only effective for "DDP", "KL", "ED", "Chi", "CTS", "Roi", "Net","Gmv" methods.

    control_name: string
        The name of the control group (other experiment groups will be regarded as treatment groups)

    normalization: boolean, optional (default=True)
        The normalization factor defined in Rzepakowski et al. 2012, correcting for tests with large number of splits
        and imbalanced treatment and control splits

    """

    def __init__(self, max_features=None, max_depth=3, min_samples_leaf=100,
                 min_samples_treatment=10, n_reg=100, evaluationFunction='KL', norm=True,
                 control_name=None, normalization=False, x_names=None,if_prune=False):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_treatment = min_samples_treatment
        self.n_reg = n_reg
        self.max_features = max_features
        self.x_names = x_names
        self.if_prune = if_prune
        self.norm = norm

        assert evaluationFunction in ["DDP", "KL", "ED", "Chi", "CTS", "Roi","Gmv",
                                      "Net", "MaxGmv"],\
            'evaluation function must be in ["DDP", "KL", "ED", "Chi", "CTS", "Roi", "Net","Gmv","MaxGmv"]'

        self.evaluationFunctionName = evaluationFunction

        evalute_dict = {'KL': evaluate_KL, 'ED': evaluate_ED, 'Chi': evaluate_Chi, 'CTS': evaluate_CTS,
                        'Net': evaluate_Net, 'Gmv': evaluate_Gmv,
                        'DDP': evaluate_DDP, 'MaxGmv': evaluate_Max_Gmv}

        self.evaluationFunction = evalute_dict[evaluationFunction]

        self.fitted_uplift_tree = None
        self.control_name = control_name
        self.normalization = normalization

    @staticmethod
    def kl_divergence(pk, qk):
        '''
         考虑了对称性
        Calculate KL Divergence for binary classification.

        sum(np.array(pk) * np.log(np.array(pk) / np.array(qk)))

        Args
        ----
        pk : float
            The probability of 1 in one distribution.
        qk : float
            The probability of 1 in the other distribution.

        Returns
        -------
        S : float
            The KL divergence.
        '''
        if qk < 0.1 ** 6:
            qk = 0.1 ** 6
        elif qk > 1 - 0.1 ** 6:
            qk = 1 - 0.1 ** 6
        # 某个treatment下分为0和为1两种情况考虑
        # S = pk * np.log(pk / qk) + (1 - pk) * np.log((1 - pk) / (1 - qk))
        S = pk * np.log(pk / qk)
        return S

    @staticmethod
    def entropyH(p, q=None):
        '''
        Entropy

        Entropy calculation for normalization.

        Args
        ----
        p : float
            The probability used in the entropy calculation.

        q : float, optional, (default = None)
            The second probability used in the entropy calculation.

        Returns
        -------
        entropy : float
        '''
        # print("p = ", p)
        # if q is None and p > 0:
        #     return -p * np.log(p)
        # elif q > 0:
        #     return -p * np.log(q)
        # else:
        #     return 0
        if q is None and p > 0:
            return -p * np.log(p)
        elif p < 0:
            return -p * np.log(q)
        else:
            return 0
    
    def normI(self, currentNodeSummary, leftNodeSummary, rightNodeSummary, control_name, alpha=0.9):
        '''
        # Current Node Info and Summary  {'control':[mean_gmv, sample, mean_cost, gmv, cost, sigma_gmv, distribution]}

        Normalization factor.
        Returns
        -------
        norm_res : float
            Normalization factor.
        '''
        norm_res = 0
        # n_t, n_c: gmv for all treatment, and control
        # pt_a, pc_a: % of treatment is in left node, % of control is in left node
        n_c = currentNodeSummary[control_name][3]
        n_c_left = leftNodeSummary[control_name][3]
        n_t = []
        n_t_left = []
        for treatment_group in currentNodeSummary:
            if treatment_group != control_name:
                n_t.append(currentNodeSummary[treatment_group][3])
                if treatment_group in leftNodeSummary:
                    n_t_left.append(leftNodeSummary[treatment_group][3])
                else:
                    n_t_left.append(0)
        pt_a = 1. * np.sum(n_t_left) / (np.sum(n_t) + 0.1)
        pc_a = 1. * n_c_left / (n_c + 0.1)
        for i in range(len(n_t)):
            pt_a_i = 1. * n_t_left[i] / (n_t[i] + 0.1)
            # print("第156行 ： ", i, " : pt_a_i = ", pt_a_i)
            norm_res += (1. * n_t[i] / (np.sum(n_t) + n_c) * self.entropyH(pt_a_i))
            
        norm_res += 1. * n_c / (np.sum(n_t) + n_c) * self.entropyH(pc_a)

        norm_res += 0.5
        return norm_res
    
    def fit(self, X, treatment, y, c):
        """ Fit the uplift model.

        Args
        ----
        X : ndarray, shape = [num_samples, num_features]
            An ndarray of the covariates used to train the uplift model.

        treatment : array-like, shape = [num_samples]
            An array containing the treatment group for each unit.

        y : array-like, shape = [num_samples]
            An array containing the outcome of interest for each unit.

        c : array-like ,shape = [num_samples]
            An array containing the outcome of cost for each unit.

        Returns
        -------
        self : object
        """
        assert len(X) == len(y) and len(X) == len(treatment), 'Data length must be equal for X, treatment, and y.'

        # 组织特征 ，变成 features + treat + gmv + cost
        rows = [list(X[i]) + [treatment[i]] + [y[i]] + [c[i]] for i in range(len(X))]

        # 构建树模型
        resTree = self.growDecisionTreeFrom(
            rows, evaluationFunction=self.evaluationFunction,
            max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
            depth=1, min_samples_treatment=self.min_samples_treatment,
            n_reg=self.n_reg, parentNodeSummary=None
        )
        self.fitted_uplift_tree = resTree
        return self.fitted_uplift_tree


    def predict(self, X, full_output=False):
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

        '''
        p_hat_optimal = []
        treatment_optimal = []
        pred_nodes = {}
        upliftScores = []
        for xi in range(len(X)):
            pred_leaf, upliftScore = self.classify(X[xi], self.fitted_uplift_tree, dataMissing=False)
            # Predict under uplift optimal treatment
            opt_treat = max(pred_leaf, key=pred_leaf.get)
            p_hat_optimal.append(pred_leaf[opt_treat])
            treatment_optimal.append(opt_treat)
            if full_output:
                if xi == 0:
                    for key_i in pred_leaf:
                        pred_nodes[key_i] = [pred_leaf[key_i]]
                else:
                    for key_i in pred_leaf:
                        pred_nodes[key_i].append(pred_leaf[key_i])
                upliftScores.append(upliftScore)
        if full_output:
            return treatment_optimal, p_hat_optimal, upliftScores, pred_nodes
        else:
            return treatment_optimal, p_hat_optimal

    def divideSet(self, rows, column, value):
        '''
        Tree node split.

        Args
        ----

        rows : list of list
               The internal data format.

        column : int
                The column used to split the data.

        value : float or int
                The value in the column for splitting the data.

        Returns
        -------
        (list1, list2) : list of list
                The left node (list of data) and the right node (list of data).
        '''

        # for int and float values
        if isinstance(value, int) or isinstance(value, float):
            splittingFunction = lambda row: row[column] >= value
        else:  # for strings
            splittingFunction = lambda row: row[column] == value
        list1 = [row for row in rows if splittingFunction(row)]
        list2 = [row for row in rows if not splittingFunction(row)]
        return (list1, list2)

    def group_uniqueCounts(self, rows):
        '''
        Count sample size by experiment group.

        Args
        ----

        rows : list of list
               The internal data format.

        Returns
        -------
        results : dictionary
                The control and treatment : [ sample size,gmv, cost]
        '''
        results = {}
        for row in rows:
            r = row[-3]
            y = row[-2]
            c = row[-1]
            if r not in results:
                results[r] = [0, 0, 0]

            results[r][0] += 1
            results[r][1] += y
            results[r][2] += c

        return results

    def tree_node_summary(self, rows, min_samples_treatment=10, n_reg=100, parentNodeSummary=None):
        '''
        Tree node summary statistics.
        计算一个节点 的 control 和treatment  指标情况

        Args
        ----

        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).
            row[-1]是成本 ， row[-1]是gmv ， row[-3]是treatment
        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group t be split at a leaf node.

        n_reg :  int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.

        parentNodeSummary : dictionary
            Node summary statistics of the parent tree node.

        Returns
        -------
        nodeSummary : dictionary
            The node summary of the current tree node.
        '''
        # returns {treatment_group: p(1)}

        yValues = [row[-2] for row in rows]
        lspercentile = np.percentile(yValues, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        lsUnique = list(set(lspercentile))
        lsUnique.sort()
        #测试
        # if parentNodeSummary is None:
        #     print("yValues : ", yValues)
        #     print("lspercentile : ", lspercentile)
        #     print("lsUnique : ", lsUnique)

        if lsUnique[0] == 0:
            lsUnique[0] = 0.00001

        nodeSummary = {}
        for row in rows:
            r = row[-3]
            y = row[-2]
            c = row[-1]
            if r not in nodeSummary:
                nodeSummary[r] = [0, 0, 0, 0, [0] * (len(lsUnique) + 1)]

            nodeSummary[r][0] += 1  # node
            nodeSummary[r][1] += y  # gmv sum
            nodeSummary[r][2] += c  # cost sum
            nodeSummary[r][3] += y * y

            bucket_index = len(lsUnique)
            for i, val in enumerate(lsUnique):
                if y < val:
                    bucket_index = i
                    break
            nodeSummary[r][4][bucket_index] += 1 #gmv 分桶后计数

        res = {}
        for r in nodeSummary:
            sample = nodeSummary[r][0]
            gmv = nodeSummary[r][1]
            cost = nodeSummary[r][2]
            gmv_square = nodeSummary[r][3]
            distribution = nodeSummary[r][4]
            mean_gmv = gmv / sample
            mean_cost = cost / sample
            mean_gmv_square = gmv_square / sample
            sigma_gmv = mean_gmv_square - mean_gmv ** 2
            res[r] = [mean_gmv, sample, mean_cost, gmv, cost, sigma_gmv, distribution]
        
        # print("res : ", res)
        return res

    def uplift_classification_results(self, rows):

        '''
        Classification probability for each treatment in the tree node.
        Args
        ----
        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

        Returns
        -------
        res : dictionary
            The probability of 1 in each treatment in the tree node.
        '''

        results = self.group_uniqueCounts(rows) # [sample, gmv ,cost]
        res = {}
        for r in results:
            if r == self.control_name:
                res[r] = 0.0
            else:
                t_gmv = float(results[r][1])
                t_cost = float(results[r][2])
                c_gmv = float(results[self.control_name][1])
                c_cost = float(results[self.control_name][2])
                avg_t_gmv = t_gmv/results[r][0]
                avg_t_cost = t_cost/results[r][0]
                avg_c_gmv = c_gmv/float(results[self.control_name][0])
                avg_c_cost = c_cost/float(results[self.control_name][0])

                if t_gmv - c_gmv == 0:
                    res[r] = 0
                else :
                    if t_cost - c_cost != 0 and self.evaluationFunctionName == 'Roi':
                        roi = (t_gmv - c_gmv) / (t_cost - c_cost)
                        res[r] = round(roi, 4)
                    elif self.evaluationFunctionName == 'Gmv':
                        res[r] = round((avg_t_gmv - avg_c_gmv) / avg_c_gmv, 4)
                    else:
                        print(f't_gmv {t_gmv}, c_gmv {c_gmv} ,t_cost {t_cost} ,c_cost {c_cost}')
                        res[r] = round(0, 4)
        return res

    def growDecisionTreeFrom(self, rows, evaluationFunction, max_depth=10,
                             min_samples_leaf=100, depth=1,
                             min_samples_treatment=10, n_reg=100,
                             parentNodeSummary=None):
        '''
        Train the uplift decision tree.

        Args
        ----

        rows : list of list
            The internal data format for the training data (combining X, Y, treatment).

        evaluationFunction : string
            Choose from one of the models: 'KL', 'ED', 'Chi', 'CTS'.

        max_depth: int, optional (default=10)
            The maximum depth of the tree.

        min_samples_leaf: int, optional (default=100)
            The minimum number of samples required to be split at a leaf node.

        depth : int, optional (default = 1)
            The current depth.

        min_samples_treatment: int, optional (default=10)
            The minimum number of samples required of the experiment group to be split at a leaf node.

        n_reg: int, optional (default=10)
            The regularization parameter defined in Rzepakowski et al. 2012,
            the weight (in terms of sample size) of the parent node influence
            on the child node, only effective for 'KL', 'ED', 'Chi', 'CTS' methods.

        parentNodeSummary : dictionary, optional (default = None)
            Node summary statistics of the parent tree node.

        Returns
        -------
        object of DecisionTree class
        '''

        if len(rows) == 0:
            return UpliftTree()

        # Current Node Info and Summary  {'control':[mean_gmv, sample, mean_cost, gmv, cost, sigma_gmv, distribution]}
        currentNodeSummary = self.tree_node_summary(
            rows, min_samples_treatment=min_samples_treatment, n_reg=n_reg, parentNodeSummary=parentNodeSummary
        )
        # 计算根节点score
        currentScore = evaluationFunction(currentNodeSummary, control_name=self.control_name)

        # Prune Stats
        maxAbsDiff = 0
        maxDiff = -10000000.
        bestTreatment = self.control_name
        maxDiffTreatment = self.control_name
        maxDiffSign = 0

        # 叶子节点的权重是分裂增益带来的 ，是同一个节点内部gmv差异,每个叶子节点存了一个最好的treatment 和  diff值
        for treatment_group in currentNodeSummary:
            if treatment_group != self.control_name:
                diff = currentNodeSummary[treatment_group][0] - currentNodeSummary[self.control_name][0]
                if abs(diff) >= maxAbsDiff:
                    maxDiffTreatment = treatment_group
                    maxDiffSign = np.sign(diff)
                    maxAbsDiff = abs(diff)
                if diff >= maxDiff:
                    maxDiff = diff
                    if diff > 0:
                        bestTreatment = treatment_group

        sigma_t = currentNodeSummary[bestTreatment][5]
        n_t = currentNodeSummary[bestTreatment][1]
        sigma_c = currentNodeSummary[self.control_name][5]
        n_c = currentNodeSummary[self.control_name][1]
        p_value = 1.96 * np.sqrt(sigma_t / n_t + sigma_c / n_c)

        upliftScore = [maxDiff, p_value]

        bestGain = 0.0
        bestAttribute = None
        bestSets = None

        # find best feature for split
        columnCount = len(rows[0]) - 3
        # if (self.max_features and self.max_features > 0 and self.max_features <= columnCount):
        #     max_features = self.max_features
        # else:
        #     max_features = columnCount
        print('columnCount: ', columnCount)
        print('columnCount*self.max_features: ', columnCount*self.max_features)

        # randomCols = list(np.random.choice(a=range(columnCount), size=max_features, replace=False))
        randomCols = list(np.random.choice(a=range(columnCount), size=int(columnCount*self.max_features), replace=False))
        randomCols.sort()
        # print("第", depth, "层采样的随机特征: ", list(map(lambda x: self.x_names[x], randomCols)))

        for col in tqdm(randomCols):

            columnValues = [row[col] for row in rows]
            # unique values
            lsUnique = list(set(columnValues))

            if (isinstance(lsUnique[0], int) or
                    isinstance(lsUnique[0], float)):
                if len(lsUnique) > 10:
                    lspercentile = np.percentile(columnValues, [3, 5, 10, 20, 30, 50, 70, 80, 90, 95, 97])
                else:
                    lspercentile = np.percentile(lsUnique, [10, 50, 90])
                lsUnique = list(set(lspercentile))

            lsUnique.sort()
            # print("\n第", depth, "层第", col, "个【特征】", self.x_names[col], " : ", lsUnique)
            for value in lsUnique:
                (set1, set2) = self.divideSet(rows, col, value)
                # check the split validity on min_samples_leaf  372
                if (len(set1) < min_samples_leaf or len(set2) < min_samples_leaf):
                    continue
                # summarize notes
                # Gain -- Entropy or Gini

                leftNodeSummary = self.tree_node_summary(
                    set1, min_samples_treatment=min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )

                rightNodeSummary = self.tree_node_summary(
                    set2, min_samples_treatment=min_samples_treatment,
                    n_reg=n_reg, parentNodeSummary=currentNodeSummary
                )
                # check the split validity on min_samples_treatment
                if set(leftNodeSummary.keys()) != set(rightNodeSummary.keys()):
                    continue
                node_mst = 10 ** 8
                for ti in leftNodeSummary:
                    node_mst = np.min([node_mst, leftNodeSummary[ti][1]])  #sample
                    node_mst = np.min([node_mst, rightNodeSummary[ti][1]])
                    total_sample = leftNodeSummary[ti][1] + rightNodeSummary[ti][1]
                if node_mst < min_samples_treatment or total_sample/node_mst >10 :
                    continue
                # evaluate the split

                p = float(len(set1)) / len(rows)
                leftScore1 = evaluationFunction(leftNodeSummary, control_name=self.control_name)
                rightScore2 = evaluationFunction(rightNodeSummary, control_name=self.control_name)

                gain = (p * leftScore1 + (1 - p) * rightScore2 - abs(currentScore))
                norm_factor = 1.0
                # 正则
                if self.norm:
                    norm_factor = self.normI(currentNodeSummary,
                                            leftNodeSummary,
                                            rightNodeSummary,
                                            self.control_name,
                                            alpha=0.9)
                gain = gain / norm_factor

                if (gain > bestGain and len(set1) > min_samples_leaf and
                        len(set2) > min_samples_leaf):
                    bestGain = gain
                    bestAttribute = (col, value)
                    bestSets = (set1, set2)
        if bestAttribute is not None:
            print(
                    f'the depth:{depth} ,the featue name:{self.x_names[bestAttribute[0]]},'
                    f'the gain:{bestGain} the left  sample {len(bestSets[0])} the right sample {len(bestSets[1])}')

        dcY = {'impurity': '%.3f' % currentScore, 'samples': '%d' % len(rows)}
        # Add treatment size
        dcY['group_size'] = ''
        for treatment_group in currentNodeSummary:
            dcY['group_size'] += ' ' + treatment_group + ': ' + str(currentNodeSummary[treatment_group][1])
        dcY['upliftScore'] = [round(upliftScore[0], 4), round(upliftScore[1], 4)]  # diff , p_value
        dcY['matchScore'] = round(upliftScore[0], 4) # diff

        if bestGain > 0 and depth < max_depth:

            trueBranch = self.growDecisionTreeFrom(
                bestSets[0], evaluationFunction, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )
            falseBranch = self.growDecisionTreeFrom(
                bestSets[1], evaluationFunction, max_depth, min_samples_leaf,
                depth + 1, min_samples_treatment=min_samples_treatment,
                n_reg=n_reg, parentNodeSummary=currentNodeSummary
            )

            return UpliftTree(
                col=bestAttribute[0], value=bestAttribute[1],
                trueBranch=trueBranch, falseBranch=falseBranch, summary=dcY,
                maxDiffTreatment=maxDiffTreatment, maxDiffSign=maxDiffSign,
                nodeSummary=currentNodeSummary,
                backupResults=self.uplift_classification_results(rows),
                bestTreatment=bestTreatment, upliftScore=upliftScore
            )
        else:
            return UpliftTree(
                    results=self.uplift_classification_results(rows),
                    summary=dcY, maxDiffTreatment=maxDiffTreatment,
                    maxDiffSign=maxDiffSign, nodeSummary=currentNodeSummary,
                    bestTreatment=bestTreatment, upliftScore=upliftScore
                )

    def classify(self, observations, tree, dataMissing=False):
        '''
        Classifies (prediction) the observationss according to the tree.

        Args
        ----

        observations : list of list
            The internal data format for the training data (combining X, Y, treatment).

        dataMissing: boolean, optional (default = False)
            An indicator for if data are missing or not.

        Returns
        -------
        tree.results, tree.upliftScore :
            The results in the leaf node.
        '''

        def classifyWithoutMissingData(observations, tree):
            '''
            Classifies (prediction) the observationss according to the tree, assuming without missing data.

            Args
            ----

            observations : list of list
                The internal data format for the training data (combining X, Y, treatment).

            Returns
            -------
            tree.results, tree.upliftScore :
                The results in the leaf node.
            '''
            if tree.results is not None:  # leaf
                return tree.results, tree.upliftScore
            else:
                v = observations[tree.col]
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
                else:
                    if v == tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
            return classifyWithoutMissingData(observations, branch)

        return classifyWithoutMissingData(observations, tree)



if __name__ == '__main__':
    pass