from __future__ import division

import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
import drugbank_lookup
import coreference
import math
import nltk 
import BeautifulSoup as bs 
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from string import punctuation

pmid_begin = "PMIDBEGIN"
pmid_end = "PMIDEND"


def _just_the_txt(s):
    return " ".join(s.findAll(text=True)).strip()

def get_tokens_and_lbls(annotated_data_path="summerscales-annotated-abstracts", 
                            start_and_stop_tokens=True,
                            use_pickle=True, use_coref=True):
    stop_token =  "STOPSTOPSTOP"
    start_token = "STARTSTARTSTART"

    stop_words = ['a', 'an', 'of', 'the', 'and', 'in', '--', "'s", "``", "''", "n't"]

    pmids, docs, corpus_tokens, lbls = [], [], [], []
    pmids_dict = {}

    for f in _get_xml_file_names(annotated_data_path):
        soup = bs.BeautifulSoup(open(f).read())
        pmid = soup.find("abstract")['id']
        sentences = list(soup.findAll("s"))
        
        ordered_sentences = []
        ordered_sentences_clean = []
        sentence_ids = [int(sent["id"]) for sent in sentences]
        d = dict(zip(sentence_ids, sentences))

        for s_id in sorted(d.keys()):
            cur_s = d[s_id]
            # including tags
            ordered_sentences.append(cur_s)
            # stripped of tags; just the text
            ordered_sentences_clean.append(_just_the_txt(cur_s))

        if start_and_stop_tokens:
            doc_sentence_tokens, doc_sentence_lbls = [[pmid_begin+pmid+pmid_end+start_token]], [[-1]]
            abstract_tokens, abstract_lbls = [pmid_begin+pmid+pmid_end+start_token], [-1]

        else:
            doc_sentence_tokens, doc_sentence_lbls = [], []

        for sent_idx, sent in enumerate(ordered_sentences):
            cleaned_sent, group_tokenized, sent_tokenized = [None]*3 # for sanity
            cleaned_sent = ordered_sentences_clean[sent_idx]
            sent_tokenized = nltk.word_tokenize(cleaned_sent)

            #remove stop words and punctuation tokens from sentences
            sent_tokenized = [element.lower() for element in sent_tokenized if element.lower() not in stop_words and element not in punctuation]

            # initialize label vector for the current sentence to -1s
            sent_lbls = [-1 for _ in sent_tokenized]

            group_strs_in_sent = sent.findAll("group")
            for group_str in group_strs_in_sent:
                group_text = _just_the_txt(group_str) #group_str.text

                group_tokenized = nltk.word_tokenize(group_text)

                #remove stop words from groups
                group_tokenized = [element.lower() for element in group_tokenized if element.lower() not in stop_words and element not in punctuation]

                # and now flip on labels corresponding to the tokens in the
                # group span we read out
                try:
                    start_index = sent_tokenized.index(group_tokenized[0])
                except:
                    pdb.set_trace()
                group_len = len(group_tokenized)
                sent_lbls[start_index:start_index+group_len] = [1]*group_len


            #prepend PMID to token
            prefix = pmid_begin + pmid + pmid_end
            sent_tokenized = [prefix + element.lower() for element in sent_tokenized]

            doc_sentence_lbls.append(sent_lbls)
            doc_sentence_tokens.append(sent_tokenized)
            abstract_lbls.extend(sent_lbls)
            abstract_tokens.extend(sent_tokenized)

        if start_and_stop_tokens:
            doc_sentence_lbls.append([-1])
            doc_sentence_tokens.append([pmid_begin+pmid+pmid_end+stop_token])
            abstract_lbls.extend([-1])
            abstract_tokens.extend([pmid_begin+pmid+pmid_end+stop_token])

        pmids_dict[pmid] = (doc_sentence_tokens, doc_sentence_lbls)

        corpus_tokens.extend(abstract_tokens)

        docs.extend(doc_sentence_tokens)
        lbls.extend(doc_sentence_lbls)
        pmids.extend([pmid]*len(doc_sentence_lbls))


    if use_pickle:
        if use_coref:
            token_to_features = pickle.load( open("token_to_features_coref_tfidf.p", "rb"))
        else:
            token_to_features = pickle.load( open("token_to_features_no_coref.p", "rb"))
    else:
        v = CountVectorizer(ngram_range=(1,1), binary=True, lowercase=False, tokenizer=nltk.word_tokenize)

        token_to_features = {}
        drugbank = drugbank_lookup.Drugbank()

        count_features = v.fit_transform(corpus_tokens)
        count_features = count_features.toarray()

        print("tokens: ", corpus_tokens[:1], ", shape: ", len(corpus_tokens))

        _print_counts(count_features, v, False)

        dict_keys = v.vocabulary_.keys()
        num_docs = len(pmids_dict.keys())
        for token in v.vocabulary_.keys():
            pmid = get_pmid_from_token(token)
            cleaned_word = get_word_from_token(token)
            is_drug = drugbank.is_in_Drugbank(cleaned_word.lower())
            token_count = np.sum(count_features[:,v.vocabulary_[token]])
            num_matching_docs = sum([dict_key.endswith(pmid_end+cleaned_word) for dict_key in dict_keys])
            token_idf = math.log(num_docs/num_matching_docs)
            token_tfidf = token_count + token_count*token_idf
            if (use_coref):
                coref_results = coreference.get_coref_counts_for_word_in_pmid(pmid, cleaned_word)
                token_to_features[token] = [token_count, token_tfidf, is_drug, coref_results[0], coref_results[1]]
            else:
                token_to_features[token] = [token_count, token_tfidf, is_drug]
        if (use_coref):
            pickle.dump( token_to_features, open( "token_to_features_coref_tfidf.p", "wb" ) )
        else:
            pickle.dump( token_to_features, open( "token_to_features_no_coref_tfidf.p", "wb" ) )

    all_labels = []
    for label in lbls:
        all_labels.extend(label)
    print("features for Mandometer: ", token_to_features["PMIDBEGIN20051465PMIDENDmandometer"])
    print("features for group: ", token_to_features["PMIDBEGIN20051465PMIDENDgroup"])
    print("features for intervention: ", token_to_features["PMIDBEGIN20051465PMIDENDintervention"])
    print("features for standard: ", token_to_features["PMIDBEGIN20051465PMIDENDstandard"])
    print("features for eating: ", token_to_features["PMIDBEGIN20051465PMIDENDeating"])
    print("features for months: ", token_to_features["PMIDBEGIN20051465PMIDENDmonths"])
    print("features for mean: ", token_to_features["PMIDBEGIN20051465PMIDENDmean"])

    length_per_doc = []
    for pmid in pmids_dict:
        _, pmid_lbls_sents = pmids_dict[pmid]
        length_per_sent = 0
        for pmid_lbls_sent in pmid_lbls_sents:
            length_per_sent += len(pmid_lbls_sent)
        length_per_doc.append(length_per_sent)
    print("mean %s, max %s, min %s, var %s" % (np.mean(token_to_features.values(), axis=0), np.max(token_to_features.values(), axis=0), np.min(token_to_features.values(), axis=0), np.var(token_to_features.values(), axis=0)))

    print("per doc: mean %s, max %s, min %s, var %s") % (np.mean(length_per_doc), np.max(length_per_doc), np.min(length_per_doc), np.var(length_per_doc))

    return pmids_dict, token_to_features


def get_word_from_token(token):
    start_index = token.index(pmid_end) + len(pmid_end)
    return token[start_index:]

def get_pmid_from_token(token):
    start_index = token.index(pmid_begin) + len(pmid_begin)
    end_index = token.index(pmid_end)
    return token[start_index:end_index]

def plot_distributions(X1):
    fig = plt.figure()

    ax1 = fig.add_subplot(111) #row, column figure
    ax1.scatter(range(len(X1)), X1, c='r', alpha=0.5)
    plt.show()

def plot_distributions_loop(X1):
    fig = plt.figure()
    ax1 = fig.add_subplot(111) #row, column figure
    for counts_per_abstract in X1:
        ax1.scatter(counts_per_abstract, range(len(counts_per_abstract)), c='r', alpha=0.5)
    plt.show()

def plot_hist(X1, loop):
    fig = plt.figure()
    ax1 = fig.add_subplot(111) #row, column figure
    if loop:
        counts = []
        for counts_per_abstract in X1:
            counts.extend(counts_per_abstract)
        ax1.hist(counts, alpha=0.5)
    else:
        ax1.hist(X1, alpha=0.5)
    plt.show()

def _get_xml_file_names(dir_):
    return [os.path.join(dir_, f) 
        for f in os.listdir(dir_) if f.endswith(".xml")]

def _print_counts(features, vectorizer, is_sparse):
    # Sum up the counts of each vocabulary word (sum of each column)
    if is_sparse:
        dist = features.sum(axis=0)
    else:
        dist = np.sum(features, axis=0)
    print "Dimensions: ", features.shape
    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    vocab = vectorizer.get_feature_names()
    sorted_counts = sorted(zip(vocab, dist), key=lambda count: count[1], reverse=True)
    print "Counts: ", sorted_counts[:40]