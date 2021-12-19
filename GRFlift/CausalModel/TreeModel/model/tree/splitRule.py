import numpy as np


def kl_divergence(pk, qk):
    '''
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
    S = pk * np.log(pk / qk) + (1 - pk) * np.log((1 - pk) / (1 - qk))
    return S


def evaluate_KL(nodeSummary, control_name):
    '''
    Calculate KL Divergence as split evaluation criterion for a given node.

    Args
    ----

    nodeSummary : dictionary
        The tree node summary statistics, produced by tree_node_summary()
        method.

    control_name : string
        The control group name.

    Returns
    -------
    d_res : KL Divergence
    '''
    if control_name not in nodeSummary:
        return 0
    c_distribution = nodeSummary[control_name][6] / float(nodeSummary[control_name][1])
    d_res = 0
    for treatment_group in nodeSummary:
        if treatment_group != control_name:
            t_distribution = nodeSummary[treatment_group][6] / float(nodeSummary[treatment_group][1])
            for i in range(len(t_distribution)):
                t = t_distribution[i] if t_distribution[i] > 0 else 0.1 ** 6
                c = c_distribution[i] if c_distribution[i] > 0 else 0.1 ** 6
                d_res += t * np.log(t / c)
    return d_res


def evaluate_ED(nodeSummary, control_name):
    '''
    Calculate Euclidean Distance as split evaluation criterion for a given node.
    Args
    ----

    nodeSummary : dictionary
        The tree node summary statistics, produced by tree_node_summary()
        method.

    control_name : string
        The control group name.

    Returns
    -------
    d_res : Euclidean Distance
    '''
    if control_name not in nodeSummary:
        return 0
    c_distribution = nodeSummary[control_name][6]
    d_res = 0
    for treatment_group in nodeSummary:
        if treatment_group != control_name:
            t_distribution = nodeSummary[control_name][6]
            for i in range(len(t_distribution)):
                t = t_distribution[i] if t_distribution[i] > 0 else 0.1 ** 6
                c = c_distribution[i] if c_distribution[i] > 0 else 0.1 ** 6
                d_res += (t - c) ** 2
    return d_res


def evaluate_Chi(nodeSummary, control_name):
    '''
    Calculate Chi-Square statistic as split evaluation criterion for a given node.

    Args
    ----

    nodeSummary : dictionary
        The tree node summary statistics, produced by tree_node_summary() method.

    control_name : string
        The control group name.

    Returns
    -------
    d_res : Chi-Square
    '''
    if control_name not in nodeSummary:
        return 0
    c_distribution = nodeSummary[control_name][6]
    d_res = 0
    for treatment_group in nodeSummary:
        if treatment_group != control_name:
            t_distribution = nodeSummary[control_name][6]
            for i in range(len(t_distribution)):
                t = t_distribution[i] if t_distribution[i] > 0 else 0.1 ** 6
                c = c_distribution[i] if c_distribution[i] > 0 else 0.1 ** 6
                d_res += (t - c) ** 2 / t
    return d_res


def evaluate_CTS(currentNodeSummary):
    '''
    Calculate CTS (conditional treatment selection) as split evaluation criterion for a given node.

    Args
    ----

    nodeSummary : dictionary
        The tree node summary statistics, produced by tree_node_summary() method.

    control_name : string
        The control group name.

    Returns
    -------
    d_res : Chi-Square
    '''
    mu = 0.0
    # iterate treatment group
    for r in currentNodeSummary:
        mu = max(mu, currentNodeSummary[r][0])
    return -mu


def evaluate_Roi(nodeSummary, control_name):
    if control_name not in nodeSummary:
        return 0
    c_gmv = float(nodeSummary[control_name][0])
    c_cost = float(nodeSummary[control_name][2])

    t_gmv_total = 0
    t_cost_total = 0
    t_pas_total = 0
    for treatment_group in nodeSummary:
        if treatment_group != control_name:
            t_gmv_total += nodeSummary[treatment_group][3]
            t_cost_total += nodeSummary[treatment_group][4]
            t_pas_total += nodeSummary[treatment_group][1]

    t_gmv = float(t_gmv_total / t_pas_total)
    t_cost = float(t_cost_total / t_pas_total)

    if t_cost - c_cost <= 0:
        return 0.0

    d_res = (t_gmv - c_gmv) / (t_cost - c_cost)
    return d_res


def evaluate_Gmv(nodeSummary, control_name):

    """
     node Summary [mean_gmv, sample, mean_cost, gmv, cost, sigma_gmv, distribution]
     Mae gmv;
    :param control_name:
    :return:
    """
    if control_name not in nodeSummary:
        return 0
    c_gmv = float(nodeSummary[control_name][0])

    t_gmv_total = 0
    t_cost_total = 0
    t_pas_total = 0

    for treatment_group in nodeSummary:
        if treatment_group != control_name:
            t_gmv_total += nodeSummary[treatment_group][3]
            t_cost_total += nodeSummary[treatment_group][4]
            t_pas_total += nodeSummary[treatment_group][1]

    t_gmv = float(t_gmv_total / t_pas_total)
    d_res = abs(t_gmv - c_gmv)
    return d_res




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
    if q is None and p > 0:
        return -p * np.log(p)
    elif q > 0:
        return -p * np.log(q)
    else:
        return 0


