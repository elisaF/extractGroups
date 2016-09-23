__author__ = 'elisa'

import BeautifulSoup as bs
import matplotlib.pyplot as plt
import os 
import nltk

pmid_to_corefs = {}
coref_results = {}
def _get_xml_file_names(dir_):
    #return [os.path.join(dir_, f) 
    return [f for f in os.listdir(dir_) if f.endswith(".xml")]

def build_coreference_dict(coreference_data_path='/Users/elisa/Documents/CompLing/stanford-corenlp-full-2015-12-09/coref/'):
    file_names = _get_xml_file_names(coreference_data_path)
    global pmid_to_corefs
            
    for file_name in file_names:
        pmid = file_name[:file_name.find('.')]
        soup = bs.BeautifulSoup(open(os.path.join(coreference_data_path, file_name)).read())
        coref_chain_to_texts = []
        coref_chains = list(soup.findAll("coreference"))
        for coref_chain in coref_chains:
            mention_text_list = []
            mentions = list(coref_chain.findAll("mention"))
            #make sure there is at least one mention
            if mentions:
                for mention in mentions:
                    texts = list(mention.findAll("text"))
                    for text in texts:
                        tokenized_text = nltk.word_tokenize(text.string)
                        tokenized_text = [token.lower() for token in tokenized_text]
                        mention_text_list.extend(tokenized_text)
                coref_chain_to_texts.append(mention_text_list)
        pmid_to_corefs[pmid] = coref_chain_to_texts

def get_coref_counts_for_word_in_pmid(pmid, lookup_word):
    global coref_results
    lookup_key = (pmid, lookup_word)
    if lookup_key not in coref_results:
        if not pmid_to_corefs:
            build_coreference_dict()
        coref_chain_to_texts = pmid_to_corefs[pmid]
        max_number_occurences_in_chain = 0
        number_occurences_in_chain = []
        number_occurences_in_abstract = 0
        number_chains = 0
        for coref_chain in coref_chain_to_texts:
            word_in_coref_count = coref_chain.count(lookup_word)
            if word_in_coref_count:
                number_chains += 1
            number_occurences_in_chain.append(word_in_coref_count)
            number_occurences_in_abstract += word_in_coref_count
        if number_occurences_in_chain:
            max_number_occurences_in_chain = max(number_occurences_in_chain)
        coref_results[lookup_key] = [max_number_occurences_in_chain, number_chains]
    return coref_results[lookup_key]

def get_coref_counts_for_word(lookup_word):
    global pmid_to_corefs

    #only build if it's not empty
    if not pmid_to_corefs:
        build_coreference_dict()
    for pmid in pmid_to_corefs.keys():
        coref_chain_to_texts = pmid_to_corefs[pmid]
        max_number_occurences_in_chain = 0
        number_occurences_in_chain = []
        number_occurences_in_abstract = 0
        number_chains = 0
        for coref_chain in coref_chain_to_texts:
            word_in_coref_count = coref_chain.count(lookup_word)
            if word_in_coref_count:
                number_chains += 1
            number_occurences_in_chain.append(word_in_coref_count)
            number_occurences_in_abstract += word_in_coref_count
        if number_occurences_in_chain:
            max_number_occurences_in_chain = max(number_occurences_in_chain)
        coref_results[pmid] = [max_number_occurences_in_chain, number_chains]
    #plot_distributions(coref_results)
    return coref_results
    
def plot_distributions(coref_results):
    fig = plt.figure()

    ax1 = fig.add_subplot(211) #row, column figure
    #plot max number occurences
    ax1.scatter(range(len(coref_results)), [max_occur[0] for max_occur in coref_results.values()], alpha=0.5)
    #plot number chains
    ax2 = fig.add_subplot(212)
    ax2.scatter(range(len(coref_results)), [num_chains[1] for num_chains in coref_results.values()], alpha=0.5)
    plt.show()

def get_coref_counts_for_vocabulary(vocabulary):
    word_to_coref_results = {}
    for word in vocabulary:
        if (word not in word_to_coref_results):
            word_to_coref_results[word] = get_coref_counts_for_word(word)