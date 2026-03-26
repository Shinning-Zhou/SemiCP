import torch
import os
import csv
import itertools

from src import *
import CONFIG

## main experiment
def cp_margin(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'oracle'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    return results
    
def cp_class_conditional(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha, class_num):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = ClassWisePredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('class_conditional', prediction_sets, test_labels, alpha, class_num)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'oracle'
    print(f"method: {method}")
    predictor = ClassWisePredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('class_conditional', prediction_sets, test_labels, alpha, class_num)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiClasswisePredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('class_conditional', prediction_sets, test_labels, alpha, class_num)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    return results
    
def cp_group_conditional(labeled_logits, labeled_labels, labeled_groups, unlabeled_logits, unlabeled_labels, unlabeled_groups, test_logits, test_labels, test_groups, model, score, alpha, group_num):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = GroupPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, labeled_groups, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits, test_groups)
    coverage_rate, average_size, CovGap = evaluation('group_conditional', prediction_sets, test_labels, alpha, group_num=group_num, groups=test_groups)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'oracle'
    print(f"method: {method}")
    predictor = GroupPredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), torch.cat([labeled_groups, unlabeled_groups]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits, test_groups)
    coverage_rate, average_size, CovGap = evaluation('group_conditional', prediction_sets, test_labels, alpha, group_num=group_num, groups=test_groups)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiGroupPredictor(score, unlabeled_score_function, model, CONFIG.group_num, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, labeled_groups, unlabeled_logits, unlabeled_groups, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits, test_groups)
    coverage_rate, average_size, CovGap = evaluation('group_conditional', prediction_sets, test_labels, alpha, group_num=group_num, groups=test_groups)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    return results

def cp_nerghbor_num(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'oracle'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'nnm_1'
    print(f"method: {method}")
    unlabeled_score_function = UNS_n_num(n_neighbors=1)
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'nnm_2'
    print(f"method: {method}")
    unlabeled_score_function = UNS_n_num(n_neighbors=2)
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'nnm_3'
    print(f"method: {method}")
    unlabeled_score_function = UNS_n_num(n_neighbors=3)
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'nnm_5'
    print(f"method: {method}")
    unlabeled_score_function = UNS_n_num(n_neighbors=5)
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    return results    

def cp_impute_acc_test(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha, impute_acc):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'semicp'
    print(f"method: {method}")
    unlabeled_score_function = UNS_imputeacc()
    predictor = SemiPredictor_imputeacc(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, alpha, impute_acc)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    return results

def cp_ablation_nn(labeled_logits, labeled_labels, labeled_features, unlabeled_logits, unlabeled_labels, unlabeled_features, test_logits, test_labels, model, score, alpha):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'oracle'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'nnm'
    print(f"method: {method}")
    unlabeled_score_function = UNS()
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'nearest_confidence'
    print(f"method: {method}")
    unlabeled_score_function = UNS_conf()
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'nearest_score'
    print(f"method: {method}")
    unlabeled_score_function = UNSScoresNN()
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'nearest_logit'
    print(f"method: {method}")
    unlabeled_score_function = UNSLogitsNN()
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'nearest_feature'
    print(f"method: {method}")
    unlabeled_score_function = UNSFeatureNN()
    predictor = SemiPredictor_feature(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, labeled_features, unlabeled_features, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    return results

def cp_debias(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'oracle'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'naive'
    print(f"method: {method}")
    unlabeled_score_function = Naive()
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'debias'
    print(f"method: {method}")
    unlabeled_score_function = Debias()
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'random-sample'
    print(f"method: {method}")
    unlabeled_score_function = RandomSample()
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    return results

def cp_shift(shift_labeled_logits, shift_labeled_labels, labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(shift_labeled_logits, shift_labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'oracle'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(shift_labeled_logits, shift_labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    return results

def cp_cluster(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha, class_num):
    results = {}
    
    method = 'classwise'
    print(f"method: {method}")
    predictor = ClassWisePredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('class_conditional', prediction_sets, test_labels, alpha, class_num)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'classwise-oracle'
    print(f"method: {method}")
    predictor = ClassWisePredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('class_conditional', prediction_sets, test_labels, alpha, class_num)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'cluster-oracle'
    print(f"method: {method}")
    predictor = ClusteredPredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('class_conditional', prediction_sets, test_labels, alpha, class_num)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'classwise+semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiClasswisePredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('class_conditional', prediction_sets, test_labels, alpha, class_num)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'cluster'
    print(f"method: {method}")
    predictor = ClusteredPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('class_conditional', prediction_sets, test_labels, alpha, class_num)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'cluster+semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiClusteredPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('class_conditional', prediction_sets, test_labels, alpha, class_num)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    return results

def cp_interpolation(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'oracle'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(torch.cat([labeled_logits, unlabeled_logits]), torch.cat([labeled_labels, unlabeled_labels]), alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'interpolation'
    print(f"method: {method}")
    predictor = InterPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'interpolation+semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiInterPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    return results

def cp_semicp_vs_ppi(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha):
    results = {}
    
    method = 'standard'
    print(f"method: {method}")
    predictor = SplitPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'semicp'
    print(f"method: {method}")
    unlabeled_score_function = CONFIG.unlabeled_score
    predictor = SemiPredictor(score, unlabeled_score_function, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    method = 'ppi'
    print(f"method: {method}")
    predictor = PPIPredictor(score, model, temperature=1)
    predictor.calculate_threshold(labeled_logits, labeled_labels, unlabeled_logits, alpha)
    prediction_sets = predictor.predict_with_logits(test_logits)
    coverage_rate, average_size, CovGap = evaluation('margin', prediction_sets, test_labels, alpha)
    results[method] = {'coverage': coverage_rate, 'size': average_size, 'covgap': CovGap}
    
    return results


def batch_exp(CONFIG):
    device = torch.device(f"cuda:{CONFIG.cuda}" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(CONFIG.folder_path):
        os.makedirs(CONFIG.folder_path)
    with open(CONFIG.result_fname, 'a', newline='') as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=filenames)
        csv_writer.writeheader()
        
    for (dataset_name, model_name, weight), setting in itertools.product(CONFIG.dataset_model_weight, CONFIG.settings):
        # model
        model = get_model(model_name, dataset_name, device, weight)

        # dataset
        dataset = get_dataset(dataset_name, model_name, weight=weight)
        if setting in ['group_conditional', 'ablation_nn']:
            logits, labels, features = get_data_tensor(model, dataset, dataset_name=dataset_name, model_name=model_name, \
                file_dir=CONFIG.tensor_path, weight=weight, batch_size=CONFIG.batch_size, num_workers=CONFIG.num_workers, feature_extractor=True)
            features = features.to(device)
        else:
            logits, labels = get_data_tensor(model, dataset, dataset_name=dataset_name, model_name=model_name, \
                file_dir=CONFIG.tensor_path, weight=weight, batch_size=CONFIG.batch_size, num_workers=CONFIG.num_workers)

        # model evaluation
        logits = logits.to(device)
        labels = labels.to(device)
        top1 = top_logit_acc(logits, labels, top = 1)
        top5 = top_logit_acc(logits, labels, top = 5)
            
        if dataset_name == 'cifar10':
            class_num = 10
        elif dataset_name == 'cifar100':
            class_num = 100
        else:
            class_num = 1000
        
        shift_dataset_name = None
        
        for seed, labeled_num, unlabeled_num, test_num, alpha, score, impute_acc in itertools.product(CONFIG.seeds, CONFIG.avg_labeled_lens, CONFIG.unlabeled_lens, CONFIG.test_lens, CONFIG.alphas, CONFIG.scores, CONFIG.impute_accs):
            # fix randomness
            fix_randomness(seed=seed)
            
            print(f'seed: {seed}, dataset: {dataset_name}, model: {model_name}, setting: {setting}, alpha: {alpha}, score:{score.__class__.__name__}')
            if setting == 'margin': ## main experiment
                labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels = devide_semi_data(logits, labels, labeled_num, unlabeled_num, test_num)
                results = cp_margin(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha)
            elif setting == 'class_conditional': 
                labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels = devide_semi_data(logits, labels, labeled_num*class_num, unlabeled_num, test_num)
                results = cp_class_conditional(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha, class_num)
            elif setting == 'group_conditional':
                labeled_logits, labeled_labels, labeled_features, unlabeled_logits, unlabeled_labels, unlabeled_features, test_logits, test_labels, test_features = devide_semi_data(logits, labels, labeled_num*CONFIG.group_num, unlabeled_num, test_num, features=features)
                labeled_groups, centroids = cluster_feature(labeled_features, group_num=CONFIG.group_num, num_iters=CONFIG.num_iters)
                unlabeled_groups = get_groups(unlabeled_features, centroids)
                test_groups = get_groups(test_features, centroids)
                results = cp_group_conditional(labeled_logits, labeled_labels, labeled_groups, unlabeled_logits, unlabeled_labels, unlabeled_groups, test_logits, test_labels, test_groups, model, score, alpha, CONFIG.group_num)
            elif setting == 'ablation_debias':
                labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels = devide_semi_data(logits, labels, labeled_num, unlabeled_num, test_num)
                results = cp_debias(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha)
            elif setting == 'ablation_nn':
                labeled_logits, labeled_labels, labeled_features, unlabeled_logits, unlabeled_labels, unlabeled_features, test_logits, test_labels, test_features = devide_semi_data(logits, labels, labeled_num, unlabeled_num, test_num, features=features)
                results = cp_ablation_nn(labeled_logits, labeled_labels, labeled_features, unlabeled_logits, unlabeled_labels, unlabeled_features, test_logits, test_labels, model, score, alpha)
            elif setting == 'nerghbor_num':
                labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels = devide_semi_data(logits, labels, labeled_num, unlabeled_num, test_num)
                results = cp_nerghbor_num(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha)
            elif setting == 'impute_acc':
                labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels = devide_semi_data(logits, labels, labeled_num, unlabeled_num, test_num)
                results = cp_impute_acc_test(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha, impute_acc)
                if impute_acc != None:
                    top1 = impute_acc
                    top5 = None
            elif setting == 'shift':
                shift_dataset_name, shift_model_name, shift_weight = CONFIG.shift_dataset_model_weight
                shift_dataset = get_dataset(shift_dataset_name, shift_model_name, weight=shift_weight)
                shift_logits, shift_labels = get_data_tensor(model, shift_dataset, dataset_name=shift_dataset_name, model_name=shift_model_name, \
                    file_dir=CONFIG.tensor_path, weight=shift_weight, batch_size=CONFIG.batch_size, num_workers=CONFIG.num_workers)
                labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels = devide_semi_data(logits, labels, 0, unlabeled_num, test_num)
                shift_labeled_logits, shift_labeled_labels, _, _,  _, _ = devide_semi_data(shift_logits, shift_labels, labeled_num, 0, 0)
                results = cp_shift(shift_labeled_logits, shift_labeled_labels, labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha)
            elif setting == 'cp_cluster':
                labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels = devide_semi_data(logits, labels, labeled_num*class_num, unlabeled_num, test_num)
                results = cp_cluster(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha, class_num)
            elif setting == 'cp_interpolation':
                labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels = devide_semi_data(logits, labels, labeled_num, unlabeled_num, test_num)
                results = cp_interpolation(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha)
            elif setting == 'cp_semicp_ppi':
                labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels = devide_semi_data(logits, labels, labeled_num, unlabeled_num, test_num)
                results = cp_semicp_vs_ppi(labeled_logits, labeled_labels, unlabeled_logits, unlabeled_labels, test_logits, test_labels, model, score, alpha)

        
            for method in results:
                write_result_semicp(CONFIG.result_fname, seed, dataset_name, shift_dataset_name, model_name, weight, top1, top5, setting, score.__class__.__name__, CONFIG.random, method, alpha, class_num, CONFIG.group_num, results[method]['coverage'], results[method]['size'], results[method]['covgap'], labeled_num, unlabeled_num, test_num)
        

if __name__ == '__main__':
    batch_exp(CONFIG)