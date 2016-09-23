# -*- coding: utf-8 -*-

'''
Byron C. Wallace

LSTM model for picking out groups (tokens) from abstracts.
Annotated data is courtesy of Rodney Sumerscales.

Sample use:

    > import LSTM_extraction
    > LSTM_extraction.LSTM_exp() # assumes *-w2v.bin file exists!!!

Requires keras, sklearn and associated dependencies.

notes to self:

    * run in python 2.x
    * this is only groups at the moment (using LSTM)

@TODO

    * implement BoW CRFs as a baseline

'''


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import pdb
from collections import Counter

import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn import svm

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn import cross_validation
from sklearn.cross_validation import KFold

import nltk
import time
import matplotlib.pyplot as plt

import parse_summerscales
import plot_learning_curve as plc

def get_features_for_pmids(pmids_dict, pmids):
    X_tokens, X, y = [], [], []

    for pmid in pmids:
        X_token_i, X_i, y_i = pmids_dict[pmid]
        X_tokens.extend(X_token_i)
        X.extend(X_i)
        y.extend(y_i)

    X_tokens = np.vstack(X_tokens)
    X = np.vstack(X)
    y = np.hstack(y)

    print("Shapes: X_tokens: %s, X: %s, Y: %s" % (X_tokens.shape, X.shape, y.shape))

    return X, X_tokens, y

def _get_threshold_func(theta):
    def map_f(x):
        if x >= theta:
            return 1
        else:
            return -1

    vec_map_f = np.vectorize(map_f)
    return vec_map_f

def _lax_match(true_idx_star, true_tokens, pred_indices, pred_spans):
    # as per summerscales
    ignore_these = ["a", "an", "the", "of", "had", "group", "groups", "arm"]

    ###
    # any overlap?
    overlapping_indices, overlapping_tokens = None, None
    for indices, tokens in zip(pred_indices, pred_spans):
        overlapping_indices, overlapping_tokens = [], []
        for j, idx in enumerate(indices):
            if (idx in true_idx_star) and (not tokens[j] in ignore_these):
                overlapping_indices.append(idx)
                overlapping_tokens = tokens[j]
                pred_span = indices

        #overlapping_indices = [
        #    idx for j, idx in enumerate(indices) if idx in true_idx_star and
        #        not tokens[j] in ignore_these]

        if len(overlapping_indices) > 0:
            break

    if overlapping_indices is None or len(overlapping_indices) == 0:
        # no overlap
        return False

    # here, overlapping_indices refers to the subset of indices
    # in the *predicted set* that matches the indices in the
    # specified true set
    return (overlapping_indices, overlapping_tokens, pred_span)


def _evaluate_detection(y_true, y_hat, X_tokens):
    '''
    Summerscales PhD thesis, 2013
    Page 100-101

    This is the approach used for evaluating detected mentions. A detected mention is
    considered a match for an annotated mention if they consist of the same set of words
    (ignoring “a”, “an”, “the”, “of”, “had”, “group(s)”, and “arm”) or if the detected
    mention overlaps the annotated one and the overlap is not a symbol or stop
    100 word. If a detected mention overlaps multiple annotated mentions, it is
    considered to be a false positive.
    '''
    #stop_words =

    true_pos_seqs = _contiguous_pos_indices(y_true)
    pred_pos_seqs = _contiguous_pos_indices(y_hat)

    true_spans = _get_text_spans(X_tokens, true_pos_seqs)
    pred_spans = _get_text_spans(X_tokens, pred_pos_seqs)

    tps, fps = 0, 0

    tp_overlapping_tokens = []
    fp_tokens = []
    # keep track of the indices already matched
    already_matched_indices = []
    for idx, true_pos_seq in enumerate(true_pos_seqs):
        #pred_pos_seqs is zero
        matched = _lax_match(true_pos_seqs[idx], true_spans, pred_pos_seqs, pred_spans)
        if matched:
            ## overlapping indices is the set of *target* indices that
            # match the predicted tokens

            overlapping_indices, overlapping_tokens, pred_span = matched

            #print("pred span: ", pred_span, ", already matched: ", already_matched_indices)
            if not pred_span in already_matched_indices:
                already_matched_indices.append(pred_span)
                #true_pos_overlapping.append((overlapping_indices, overlapping_tokens))
                tps += 1
                tp_overlapping_tokens.append(overlapping_tokens)
            else:
                fp_tokens.append(overlapping_tokens)
                fps += 1

    ###
    # now count up predictions that were not matched
    # with any true positives
    ###
    for idx, pred_pos_seq in enumerate(pred_pos_seqs):
        # then this sequence didn't match any of the
        # true_pos_seq entries!
        if not pred_pos_seq in already_matched_indices:
            fp_tokens.append(pred_spans[idx])
            fps += 1

    if true_pos_seqs is None:
        recall = 0
    elif len(true_pos_seqs) > 0:
        recall = float(tps) / float(len(true_pos_seqs))
    else:
        recall = 0
    precision = np.divide(float(tps), float(tps + fps)) #tps can be zero!
    accuracy = accuracy_score(y_true, y_hat)
    auc = roc_auc_score(y_true, y_hat)
    return recall, precision, accuracy, auc, tp_overlapping_tokens, fp_tokens


def _contiguous_pos_indices(y):
    groups, cur_group = [], []
    last_y = None
    for idx, y_i in enumerate(list(y)):
        if y_i == last_y == 1:
            cur_group.append(idx)
        elif y_i == 1:
            # then last_y was -1, but this is 1.
            cur_group = [idx]
        elif last_y == 1:
            groups.append(cur_group)
            cur_group = []
        last_y = y_i

    if len(cur_group) > 0:
        groups.append(cur_group)
    return groups


def _get_text_spans(X_tokens, index_seqs):
    spans = []
    for idx_seq in index_seqs:
        cur_tokens = [X_tokens[idx] for idx in idx_seq]
        spans.append(cur_tokens)
    return spans

def _error_report(y_hat, y_true, X_tokens):

    true_positives = np.logical_and(np.vstack(y_hat)==1,np.vstack(y_true)==1)
    true_negatives = np.logical_and(np.vstack(y_hat)==-1,np.vstack(y_true)==-1)

    false_positives = np.logical_and(np.vstack(y_hat)==1,np.vstack(y_true)==-1)
    false_negatives = np.logical_and(np.vstack(y_hat)==-1,np.vstack(y_true)==1)

    tp_seqs = _contiguous_pos_indices(true_positives)
    tn_seqs = _contiguous_pos_indices(true_negatives)
    fp_seqs = _contiguous_pos_indices(false_positives)
    fn_seqs = _contiguous_pos_indices(false_negatives)

    tp_spans = _get_text_spans(X_tokens, tp_seqs)
    tn_spans = _get_text_spans(X_tokens, tn_seqs)
    fp_spans = _get_text_spans(X_tokens, fp_seqs)
    fn_spans = _get_text_spans(X_tokens, fn_seqs)

    print("tp %s tn %s fp %s fn %s"
          % (np.where(true_positives==1)[0].shape, np.where(true_negatives==1)[0].shape,
             np.where(false_positives==1)[0].shape, np.where(false_negatives==1)[0].shape))

    return tp_spans, tn_spans, fp_spans, fn_spans


def run(n_folds=5, use_pickle=True, use_coref=True):

    # maps pubmed identifiers to token features
    # and corresponding labels
    pmids_dict, X_tokens = get_PMIDs_to_X_y(use_pickle, use_coref)


    ''' train / test '''
    '''
        * CV on PMIDs
        *
    '''
    all_pmids = pmids_dict.keys()
    n = len(all_pmids)
    kf = KFold(n, random_state=1337, shuffle=True, n_folds=n_folds)

    title = "Learning Curves (SVM)"

    ## Learning Curve
    class_weights = {}
    class_weights[1] = 4.0462962962963
    class_weights[-1] = 0.570496083550914
    estimator = svm.SVC(class_weight=class_weights, cache_size=1000)
    train_X, _, train_y = get_features_for_pmids(pmids_dict, all_pmids)
    plc.plot_learning_curve(estimator, title, train_X, train_y, cv=5)
    plt.show()
    ##

    fold_metrics = []
    for fold_idx, (train, test) in enumerate(kf):
        print("on fold %s" % fold_idx)
        train_pmids = [all_pmids[pmid_idx] for pmid_idx in train]
        test_pmids  = [all_pmids[pmid_idx] for pmid_idx in test]
        # sanity check
        assert(len(set(train_pmids).intersection(set(test_pmids)))) == 0

        train_X, _, train_y = get_features_for_pmids(pmids_dict, train_pmids)
        test_X, test_index_X, test_y   = get_features_for_pmids(pmids_dict, test_pmids)

        #model = SGDClassifier(loss="hinge", penalty="l2", n_iter=250, alpha=0.0001, class_weight='balanced')
        class_weights = {}
        class_weights[1] = 4.0462962962963
        class_weights[-1] = 0.570496083550914
        model = svm.SVC(class_weight=class_weights, cache_size=1000)
        model.fit(train_X, train_y)

        #model = RandomForestClassifier(n_estimators = 100)
        #model.fit(train_X, train_y)

        #predict_y = list(model.predict_classes(test_X))
        predict_y = list(model.predict(test_X))
        r, p, accuracy, auc, tp_overlapping_tokens, fp_tokens = _evaluate_detection(test_y, predict_y, test_index_X)

        if p+r == 0:
            f1 = None
        else:
            f1 = (2 * p * r) / (p + r)

        tp_spans, tn_spans, fp_spans, fn_spans = _error_report(predict_y, test_y, test_index_X)

        cm = confusion_matrix(test_y, predict_y)
        np.set_printoptions(precision=2)
        print('Confusion matrix, without normalization')
        print(cm)
        plt.figure()
        plot_confusion_matrix(cm)
        # Normalize the confusion matrix by row (i.e by the number of samples
        # in each class)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
        print(cm_normalized)
        plt.figure()
        plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

        #plt.show()

        print("fold %s. precision: %s; recall: %s; f1: %s, accuracy: %s, auc: %s" % (fold_idx, p, r, f1, accuracy, auc))
        #pdb.set_trace()
        fold_metrics.append([p, r, f1, accuracy, auc])
        if use_coref:
            file_name_suffix=str(fold_idx)+'_with_coref_tfidf_'+str(time.time())+'.txt'
        else:
            file_name_suffix=str(fold_idx)+'_no_coref_tfidf_'+str(time.time())+'.txt'
        with open('results_true_'+file_name_suffix, 'wb') as results_true:
            results_true.write(str((p, r, f1, accuracy, auc))+"\n")
            results_true.write(str(tp_spans)+"\n")
            results_true.write(str(tn_spans))
        with open('results_false_'+file_name_suffix, 'wb') as results_false:
            results_false.write(str(fp_spans)+"\n")
            results_false.write(str(fn_spans))
    #convert to numpy array
    fold_metrics = np.array(fold_metrics)
    print("mean: %s, variance: %s" % (np.mean(fold_metrics, axis=0), np.var(fold_metrics, axis=0)))
    #return fold_metrics

def get_PMIDs_to_X_y(use_pickle, use_coref):
    pmids_dict, token_to_features = \
                parse_summerscales.get_tokens_and_lbls(use_pickle=use_pickle, use_coref=use_coref)

    pmids_to_X_y = {}
    for pmid in pmids_dict:
        pmid_sentences, pmid_lbls = pmids_dict[pmid]
        # for this sentence
        X_tokens, X_features = [], []
        y = []
        for sent_idx, s in enumerate(pmid_sentences):
            for j, token in enumerate(s):
                X_features.append(token_to_features[token])
                X_tokens.append(token)

            y.extend(pmid_lbls[sent_idx])

        pmids_to_X_y[pmid] = (np.vstack(X_tokens), np.vstack(X_features), np.hstack(y))
    return (pmids_to_X_y, X_tokens)

def preprocess_texts(texts, m, dim=200):

    for text in texts:
        tokenized_text = nltk.word_tokenize(text)
        for t in tokenized_text:
            try:
                v = m[t]
            except:
                # or maybe use 0s???
                v = np.random.uniform(-1,1,dim)


def plot_data(X1, X2, y):
    #get sample weight
    sample_weights = []
    c = Counter(y)

    for class_value in y:
        length = len(y)
        count = c[class_value]
        class_weight = len(y) / (2*c[class_value])
        sample_weights.append(class_weight)

    # setup figure
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    #ax1.scatter(X1, X2, c=y, cmap=plt.cm.Paired)
    ax1.scatter(X1, X2, c=y,s=sample_weights, alpha=0.9, cmap=plt.cm.Paired)
    plt.show()

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(["not_arm", "arm"]))
    plt.xticks(tick_marks, ["not_arm", "arm"], rotation=45)
    plt.yticks(tick_marks, ["not_arm", "arm"])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')