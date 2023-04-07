from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import LeaveOneOut
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from itertools import combinations
from dataclasses import dataclass
from scipy import stats
import seaborn as sns
import pingouin as pn
import pandas as pd
import numpy as np

acb_cutoff = {True: 'High', False: 'Low'}

descriptive_table_variable_order = [
    'age',
    'educ_year',
    'weight',
    'height',
    'sex',
    'dom_hnd',
    'ACB_cutoff',
    'clinical_balance_impaired',
    'clinical_fall_history',
    'biothes',
    'moca',
    'tmt_ba',
    'updrs_iii',
    'duration',
    'affected_side',
    'led_calculation',
    'striatal_dtbz'
]

numvars = [
    'age',
    'educ_year',
    'weight',
    'height',
    'biothes',
    'duration',
    'led_calculation',
    'updrs_iii',
    'moca',
    'striatal_dtbz',
    'ACB_score',
    'tmt_ba'
]

catvars = [
    'sex',
    'dom_hnd',
    'clinical_balance_impaired',
    'affected_side',
    'ACB_cutoff',
    'clinical_fall_history'
]

groupvar = 'has_fog'

var_remapper = {
    'age': 'Age (years)',
    'sex': 'Sex',
    'educ_year': 'Education (years)',
    'weight': 'Weight (kg)',
    'height': 'Height (cm)',
    'dom_hnd': 'Handedness',
    'biothes': 'Vibration Sensitivity (Hz)',
    'duration': 'PD Duration (years)',
    'led_calculation': 'LED (mg)',
    'ACB_cutoff': 'ACB Cutoff',
    'updrs_iii': 'MDS-UPDRS III Score',
    'clinical_balance_impaired': 'Clinical Balance Impairment',
    'clinical_fall_history': 'Clinical Fall History',
    'affected_side': 'More Affected Body Side',
    'moca': 'MoCA Score',
    'tmt_ba': 'Trail Making Test (B - A)',
    'striatal_dtbz': 'Striatal Dopamine PET (DTBZ)'
}

def catvar_comparison(
    df: pd.DataFrame,
    groupvar: str,
    catvar: str,
    alpha: float = 0.05,
    pvalround: int = 2
) -> tuple[str, str, str, str]:
    print('\n', catvar, '\n---')
    # MAKING CONTINGENCY TABLE
    df_contingency = pd.crosstab(df[catvar], df[groupvar])
    vec_pct = (
        df_contingency[1.0] / df_contingency.sum(axis=1) * 100
    ).round(2).astype(str).apply(lambda pct: pct+'%')
    print('\nContingency table:\n', df_contingency)
    print('\nPercentage freezers:\n', vec_pct)
    # RUNNING STATISTICS
    statistic, pval, _, _ = stats.chi2_contingency(df_contingency)
    statistic = f'χ2={np.round(statistic,2)}'
    print('\nResult:\n', statistic)
    sig = ''
    if pval < alpha:
        sig = '*'
    pval = str(np.round(pval,2)) + sig
    print('P='+pval)
    # FORMATTING DESCRIPTIVES
    group1_desc = ','.join([
        group+':'+str(value)
        for group, value in
        zip(df_contingency.index, list(df_contingency[0.0]))
    ])
    group2_desc = ','.join([
        group+':'+str(value)
        for group, value in
        zip(df_contingency.index, list(df_contingency[1.0]))
    ])
    return group1_desc, group2_desc, statistic, pval

def numvar_comparison(
    df: pd.DataFrame,
    groupvar: str,
    numvar: str,
    alpha: float = 0.05,
    pvalround: int = 2,
    descround: int = 1,
    **plotargs
) -> tuple[str, str, str, str]:
    print('\n', numvar, '\n---')
    # DESCRIPTIVES FOR GROUP 1
    sns.histplot(df, x=numvar)
    plt.show()
    vec_group1 = df.groupby(groupvar).get_group(0.0)[numvar]
    group1_median, group1_iqr = np.median(vec_group1).round(descround), stats.iqr(vec_group1).round(descround)
    group1_desc = '±'.join([str(group1_median), str(group1_iqr)])
    print(f'Non-freezers: {group1_desc}')
    # DESCRIPTIVES FOR GROUP 2
    vec_group2 = df.groupby(groupvar).get_group(1.0)[numvar]
    group2_median, group2_iqr = np.median(vec_group2).round(descround), stats.iqr(vec_group2).round(descround)
    group2_desc = '±'.join([str(group2_median), str(group2_iqr)])
    print(f'Freezers: {group2_desc}')
    # RUNNING THE TEST AND FORMATTING OUTPUTS
    result = stats.mannwhitneyu(vec_group1, vec_group2)
    statistic = f'W={np.round(result.statistic,2)}'
    print('\nResult:\n', statistic)
    sig = ''
    if result.pvalue < alpha:
        sig = '*'
    pval = str(np.round(result.pvalue,pvalround)) + sig
    print('\nP=', pval)
    # PLOTTING
    sns.boxplot(
        df,
        x=groupvar,
        y=numvar,
        fliersize=0,
        **plotargs
    )
    sns.stripplot(
        df,
        x=groupvar,
        y=numvar,
        color='gray',
        edgecolor='black',
        linewidth=1
    )
    plt.show()
    return group1_desc, group2_desc, statistic, pval

def group_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    global descriptive_table_variable_order
    global catvars, numvars, groupvar, var_remapper
    group_counts = df[groupvar].value_counts()
    results = {}
    for var in catvars:
        results[var] = catvar_comparison(df, groupvar, var, alpha=0.20)
    for var in numvars:
        results[var] = numvar_comparison(df, groupvar, var, alpha=0.20)
    df_group_comparisons = pd.DataFrame(results)[descriptive_table_variable_order].T.rename(
        var_remapper,
        axis=0
    ).rename({
        0: f'Non-freezers (N={group_counts[0]})',
        1: f'Freezers (N={group_counts[1]})',
        2: 'Statistic',
        3: 'P'
    }, axis=1)
    return df_group_comparisons


@dataclass
class LogisticModel:
    data: pd.DataFrame
    fit: LogisticRegressionCV
    coef: pd.Series
    response: str

    def predict(self, df=None, prob=True):
        if df is None:
            df = self.data
        df_regressors = df[self.coef.index]
        linpred = self.fit.intercept_ + (df_regressors @ self.coef)
        if not prob:
            return linpred
        probs = np.exp(linpred) / (1 + np.exp(linpred))
        return probs

    def benchmark_model(self):
        model_probs = self.predict()
        model_labels = self.data[self.response]
        roc = {}
        roc['fpr'], roc['tpr'], roc['thresholds'] = roc_curve(
            model_labels,
            model_probs
        )
        cutoff_index = (roc['tpr'] - roc['fpr']).argmax()
        cutoff = roc['thresholds'][cutoff_index]
        model_predictions = (model_probs > cutoff).astype(int)
        accuracy = (model_labels == model_predictions).sum() / len(model_labels)
        tn, fp, fn, tp = confusion_matrix(
            model_labels,
            model_predictions
        ).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (fp + tn)
        return LogisticBenchmark(
            roc,
            cutoff,
            accuracy,
            sensitivity,
            specificity
        )


def logistic_lasso(
    df: pd.DataFrame,
    Cs: int = 1000,
    cv = LeaveOneOut(),
    response: str = 'group',
    **args
):
    model = LogisticRegressionCV(Cs=Cs, cv=cv, penalty='l1', solver='liblinear', **args)
    model_fit = model.fit(
        X = df.drop(response, axis=1).to_numpy(),
        y = df[response].to_numpy()
    )
    model_coef = pd.Series(
        model_fit.coef_[0,:],
        index = df.drop(response, axis=1).columns
    )
    model_coef_sig = model_coef[model_coef != 0.0].sort_values()
    return LogisticModel(df, model_fit, model_coef_sig, response)


@dataclass
class LogisticBenchmark:
    roc: dict
    cutoff: float
    accuracy: float
    sensitivity: float
    specificity: float

    def __repr__(self):
        return f"""
        Cutoff: {self.cutoff}
        Accuracy: {self.accuracy}
        Sensitivity: {self.sensitivity}
        Specificity: {self.specificity}
        """

    def plot_roc_curve(self, dpi:int=150):
        fig = plt.figure(dpi=dpi)
        plt.plot(self.roc['thresholds'], self.roc['fpr'])
        plt.plot(self.roc['thresholds'], self.roc['tpr'])
        plt.legend(['False Positive', 'True Positive'])
        plt.xlabel('Decision Boundary')
        plt.ylabel('Rate')
        plt.suptitle('ROC Curve')
        plt.title(f'Cutoff: prob > {self.cutoff.round(3)}')
        plt.axvline(self.cutoff, color='red', linestyle='dashed')
        plt.xlim((0, 1))
        return fig

    def print_roc(self):
        print('Cutoff:', self.cutoff)
        print('Accuracy:', self.accuracy)
        print('Sensitivity:', self.sensitivity)
        print('Specificity:', self.specificity)


def lr_test(model_null, model_full):
    dof = len(model_full.params) - len(model_null.params)
    LR_stat = -2 * (model_null.llf - model_full.llf)
    p_val = stats.chi2.sf(LR_stat, dof)
    return LR_stat, p_val


def logistic_hierarchical_confounder_regression(
    df: pd.DataFrame,
    confounders: list[str],
    response: str = 'has_fog'
):
    predictors = []
    formula = f'{response} ~ 1'
    model = smf.logit(formula, df).fit()
    print(model.summary(), '\n\n')
    print('OR:\n', np.exp(model.params), f'\n{"".join(["-" for _ in range(25)])}\n')
    for confounder in confounders:
        model_prev = model
        predictors.append(confounder)
        formula = f'{response} ~ {" + ".join(predictors)}'
        model = smf.logit(formula, df).fit()
        print(model.summary())
        chisq, p = lr_test(model_prev, model)
        print('Chisq:', chisq)
        print('P:', p, '\n')
        print('OR:\n', np.exp(model.params), f'\n{"".join(["-" for _ in range(25)])}\n')
    return model


def pairwise_corr(df, method):
    df_corr = pd.DataFrame(
        np.zeros((df.shape[1], df.shape[1])),
        columns=df.columns,
        index=df.columns
    )
    item_combos = list(combinations(df, 2))
    for item1,item2 in item_combos:
        corr = pn.corr(df[item1], df[item2], method=method)
        df_corr.loc[item1, item1] = 1.0
        df_corr.loc[item2, item2] = 1.0
        df_corr.loc[item1, item2] = corr['r'][0]
        df_corr.loc[item2, item1] = corr['r'][0]
    return df_corr

def hierarchical_clustering(df_corr, thresh=None):
    dissimilarity = 1 - abs(df_corr)
    dissimilarity[dissimilarity == 1.0] = 0.0
    Z = linkage(squareform(dissimilarity), 'complete')
    plt.figure(dpi=150)
    if thresh is None:
        dendrogram(Z, labels=df_corr.columns, leaf_rotation=90)
        return Z
    else:
        dendrogram(Z, labels=df_corr.columns, leaf_rotation=90, color_threshold=thresh)
        plt.axhline(thresh, color='red', linestyle='dashed')
        item_clusters = pd.Series(dict(zip(
            df_corr.columns,
            fcluster(Z, thresh, criterion='distance', depth=4)
        )))
        return Z, item_clusters


def visualize_clustering(df_corr, Z, labelsize=9):
    dissimilarity = 1 - df_corr
    plt.figure(dpi=300)
    if (df_corr.min() < 0).any():
        vmin = -1
        cmap = 'bwr'
    else:
        vmin = 0
        cmap = 'viridis'
    sns.clustermap(
        1-dissimilarity, 
        method="complete", 
        cmap=cmap, 
        annot=True, 
        annot_kws={"size": labelsize}, 
        vmin=vmin, vmax=1, 
        row_linkage=Z, col_linkage=Z
    )
    plt.show()
