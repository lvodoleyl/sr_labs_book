# Exercises in order to perform laboratory work


# Import of modules
import numpy as np
from matplotlib.pyplot import hist, plot, show, grid, title, xlabel, ylabel, legend, axis, imshow


def tar_imp_hists(all_scores, all_labels):
    # Function to compute target and impostor histogram
    
    tar_scores = []
    imp_scores = []

    ###########################################################
    for score, label in zip(all_scores, all_labels):
        if label:
            tar_scores.append(score)
        else:
            imp_scores.append(score)
    
    ###########################################################
    
    tar_scores = np.array(tar_scores)
    imp_scores = np.array(imp_scores)
    
    return tar_scores, imp_scores

def llr(all_scores, all_labels, tar_scores, imp_scores, gauss_pdf):
    # Function to compute log-likelihood ratio
    
    tar_scores_mean = np.mean(tar_scores)
    tar_scores_std  = np.std(tar_scores)
    imp_scores_mean = np.mean(imp_scores)
    imp_scores_std  = np.std(imp_scores)
    
    all_scores_sort   = np.zeros(len(all_scores))
    ground_truth_sort = np.zeros(len(all_scores), dtype='bool')
    
    ###########################################################
    # Here is your code
    for_sort = list(zip(list(all_scores), list(all_labels)))
    for_sort.sort(key=lambda x: x[0])
    all_scores_sort = np.array([x[0] for x in for_sort])
    ground_truth_sort = np.array([x[1] for x in for_sort], dtype='bool')
    
    ###########################################################
    
    tar_gauss_pdf = np.zeros(len(all_scores))
    imp_gauss_pdf = np.zeros(len(all_scores))
    LLR           = np.zeros(len(all_scores))
    
    ###########################################################
    # Here is your code
    tar_gauss_pdf = gauss_pdf(all_scores_sort, tar_scores_mean, tar_scores_std)
    imp_gauss_pdf = gauss_pdf(all_scores_sort, imp_scores_mean, imp_scores_std)
    LLR = np.log(tar_gauss_pdf / imp_gauss_pdf)
    
    ###########################################################
    
    return ground_truth_sort, all_scores_sort, tar_gauss_pdf, imp_gauss_pdf, LLR

def map_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar):
    # Function to perform maximum a posteriori test
    
    len_thr = len(LLR)
    fnr_thr = np.zeros(len_thr)
    fpr_thr = np.zeros(len_thr)
    P_err   = np.zeros(len_thr)
    
    for idx in range(len_thr):
        solution = LLR > LLR[idx]                                      # decision
        
        err = (solution != ground_truth_sort)                          # error vector
        
        fnr_thr[idx] = np.sum(err[ ground_truth_sort])/len(tar_scores) # prob. of Type I  error P(Dimp|Htar), false negative rate (FNR)
        fpr_thr[idx] = np.sum(err[~ground_truth_sort])/len(imp_scores) # prob. of Type II error P(Dtar|Himp), false positive rate (FPR)
        
        P_err[idx]   = fnr_thr[idx]*P_Htar + fpr_thr[idx]*(1 - P_Htar) # prob. of error
    
    # Plot error's prob.
    plot(LLR, P_err, color='blue')
    xlabel('$LLR$'); ylabel('$P_e$'); title('Probability of error'); grid(); show()
        
    P_err_idx = np.argmin(P_err) # argmin of error's prob.
    P_err_min = fnr_thr[P_err_idx]*P_Htar + fpr_thr[P_err_idx]*(1 - P_Htar)
    
    return LLR[P_err_idx], fnr_thr[P_err_idx], fpr_thr[P_err_idx], P_err_min

def neyman_pearson_test(ground_truth_sort, LLR, tar_scores, imp_scores, fnr):
    # Function to perform Neyman-Pearson test
    
    thr   = np.inf
    fpr   = np.inf
    
    ###########################################################
    for idx in range(len(LLR)):
        solution = LLR > LLR[idx]
        err = (solution != ground_truth_sort)
        local_fpr = np.sum(err[~ground_truth_sort])/len(imp_scores)
        local_fnr = np.sum(err[ ground_truth_sort])/len(tar_scores)
        if local_fpr < fpr and local_fnr <= fnr:
            fpr = local_fpr
            thr = LLR[idx]

    ###########################################################
    
    return thr, fpr

def bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar, C00, C10, C01, C11):
    # Function to perform Bayes' test
    
    thr   = 0.0
    fnr   = 0.0
    fpr   = 0.0
    AC    = 0.0
    
    ###########################################################
    # Here is your code
    p_h0 = P_Htar
    p_h1 = 1 - p_h0
    thr = np.log(((C01 - C11)*p_h1)/((C10 - C00)*p_h0))

    solution = LLR > thr
    err = (solution != ground_truth_sort)
    fpr = np.sum(err[~ground_truth_sort]) / len(imp_scores)
    fnr = np.sum(err[ground_truth_sort]) / len(tar_scores)
    AC = C10*fpr*p_h0 + C01*fnr*p_h1
    
    ###########################################################
    
    return thr, fnr, fpr, AC

def minmax_test(ground_truth_sort, LLR, tar_scores, imp_scores, P_Htar_thr, C00, C10, C01, C11):
    # Function to perform minimax test
    
    thr    = 0.0
    fnr    = 0.0
    fpr    = 0.0
    AC     = np.inf
    P_Htar = 0.0
    
    ###########################################################
    for p_h in P_Htar_thr:
        if p_h == 0:
            continue
        thr_, fnr_, fpr_, AC_ = bayes_test(ground_truth_sort, LLR, tar_scores, imp_scores, p_h, C00, C10, C01, C11)
        if AC == np.inf or AC_ <= AC:
            thr, fnr, fpr, AC = thr_, fnr_, fpr_, AC_
            P_Htar = p_h
    
    ###########################################################
    
    return thr, fnr, fpr, AC, P_Htar