class UpliftTree:
    """ Tree Node Class

    Tree node class to contain all the statistics of the tree node.

    Parameters
    ----------

    col : int, optional (default = -1)
        The column index for splitting the tree node to children nodes.

    value : float, optional (default = None)
        The value of the feature column to split the tree node to children nodes.

    trueBranch : object of UpliftTree
        The true branch tree node (feature > value).

    falseBranch : object of UpliftTree
        The flase branch tree node (feature < value).

    results : dictionary
        The classification probability Pr(1) for each experiment group in the tree node.

    summary : dictionary
        Summary statistics of the tree nodes, including impurity, sample size, uplift score, etc.

    maxDiffTreatment : string
        The treatment name generating the maximum difference between treatment and control group.

    maxDiffSign : float
        The sign of the maxium difference (1. or -1.).

    nodeSummary : dictionary
        Summary statistics of the tree nodes {treatment: [y_mean, n]}, where y_mean stands for the target metric mean
        and n is the sample size.

    backupResults : dictionary
        The conversion proabilities in each treatment in the parent node {treatment: y_mean}. The parent node
        information is served as a backup for the children node, in case no valid statistics can be calculated from the
        children node, the parent node information will be used in certain cases.

    bestTreatment : string
        The treatment name providing the best uplift (treatment effect).

    upliftScore : list
        The uplift score of this node: [max_Diff, p_value], where max_Diff stands for the maxium treatment effect, and
        p_value stands for the p_value of the treatment effect.

    matchScore : float
        The uplift score by filling a trained tree with validation dataset or testing dataset.

    """

    def __init__(self, col=-1, value=None, trueBranch=None, falseBranch=None,
                 results=None, summary=None, maxDiffTreatment=None,
                 maxDiffSign=1., nodeSummary=None, backupResults=None,
                 bestTreatment=None, upliftScore=None, matchScore=None):
        self.col = col
        self.value = value
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.results = results  # None for nodes, not None for leaves
        self.summary = summary
        # the treatment with max( |p(y|treatment) - p(y|control)| )
        self.maxDiffTreatment = maxDiffTreatment
        # the sign for p(y|maxDiffTreatment) - p(y|control)
        self.maxDiffSign = maxDiffSign
        self.nodeSummary = nodeSummary
        self.backupResults = backupResults
        self.bestTreatment = bestTreatment
        self.upliftScore = upliftScore
        # match actual treatment for validation and testing
        self.matchScore = matchScore
