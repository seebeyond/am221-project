#!/usr/bin/python

import argparse
from collections import Counter
from constants import *
import csv
import json
import numpy as np
import random
import re
import subprocess
import time
import sys
import Stemmer  # PyStemmer - https://pypi.python.org/pypi/PyStemmer/1.0.1


#########################################################################################
# Make predictions for 1-vs-all strategy
#########################################################################################
def predict_onevsall(text, score, num, verbose):
    print "Predicting", num, "reviews"
    start = time.time()

    # import SVM dict
    svm_dict = {}
    in_file_idx = svm_folder + 'svm-dict.csv'
    f = open(in_file_idx)
    for key, val in csv.reader(f):
        svm_dict[key] = int(val)
    f.close()
    nf = len(svm_dict)
    print ' There are %d features' % nf

    # initialize lists for SVMs
    w_list = [np.zeros(nf) for i in range(ns)]
    b_list = [np.zeros(1) for i in range(ns)]
    conf_scores = np.zeros(ns)

    # counters
    error = 0
    numcorrect = 0
    
    # get random sample to predict on
    nr = len(text)    
    predict_idxs = random.sample(range(nr),num)

    # initialize stemmer
    stemmer = Stemmer.Stemmer('english')

    for k in range(num):
        # get random index
        pidx = predict_idxs[k]

        # construct feature vector for the test
        x = np.zeros(nf)
        scr = int(score[pidx]) - 1
        # get rid of escaped characters and apostrophes
        txt = text[pidx].lower().replace("&quot;","'").replace("'","")
        # reg-ex to split on
        words = re.split("[\W\s\!+\.+,+\?+\"]", txt)
        words = filter(bool, words)
        words = stemmer.stemWords(words)
        for word in words:
#             word = correct(word)     # extract root of word
            if word in svm_dict.keys():
        #         print word
                x[svm_dict[word]] = 1

        for i in range(ns):
            # SVM files
            in_file_w = svm_folder + str(i+1) + '-svm-w.txt'
            in_file_b = svm_folder + str(i+1) + '-svm-b.txt'
            # import SVM and save
            w = np.loadtxt(in_file_w)
            b = np.loadtxt(in_file_b)
            w_list[i] = w
            b_list[i] = b
            
            # get confidence score for i-vs-all SVM
#             print "length of w", len(w)
#             print "length of x", len(x)
            c = np.dot(w,x) - b
            conf_scores[i] = c

        # Print prediction and data
        predict = np.argmax(conf_scores)

        if predict == scr:
            numcorrect += 1
        error += (predict - scr) * (predict - scr)

        if verbose:
            print "Predicting", k + 1, "of", num
            print ' Predicted score is %d' % (predict + 1)
            print ' Actual score is %d' % (scr + 1)
            print ' Confidence scores:', conf_scores
            print ' Review text:'
            print ' ' + text[pidx]
            print

    end = time.time()

    print "Percentage Correct:", numcorrect / float(num)
    print "Mean Square Error:", error / float(num)

    print ' Time elapsed: %.4f sec per review' % ((end - start)/float(num))


#########################################################################################
# Make predictions for pairwise strategy
#########################################################################################
def predict_pair(text, score, num, verbose):
    print "Predicting", num, "reviews"
    start = time.time()

    # import SVM dictionary
    svm_dict = {}
    in_file_idx = svm_folder + 'svm-dict.csv'
    f = open(in_file_idx)
    for key, val in csv.reader(f):
        svm_dict[key] = int(val)
    f.close()
    nf = len(svm_dict)

    # initialize lists for SVMs
    w_list = [np.zeros(nf) for i in range(num_svms)]
    b_list = [np.zeros(1) for i in range(num_svms)]

    # counters
    error = 0
    numcorrect = 0
    
    # get random sample to predict on
    nr = len(text)    
    predict_idxs = random.sample(range(nr),num)

    # initialize stemmer
    stemmer = Stemmer.Stemmer('english')

    for k in range(num):
        # initialize
        votes = np.zeros(ns)
        svm_results = np.zeros(num_svms)
        
        # get random index
        pidx = predict_idxs[k]

        # construct feature vector for the test
        x = np.zeros(nf)
        scr = int(score[pidx]) - 1
        # get rid of escaped characters and apostrophes
        txt = text[pidx].lower().replace("&quot;","'").replace("'","")
        # reg-ex to split on
        words = re.split("[\W\s\!+\.+,+\?+\"]", txt)
        words = filter(bool, words)
        words = stemmer.stemWords(words)
        for word in words:
            if word in svm_dict.keys():
        #         print word
                x[svm_dict[word]] = 1


        for i in range(num_svms):
            # SVM files
            lower = svm_pairs[i][0] - 1
            higher = svm_pairs[i][1] - 1
            in_file_w = svm_folder + str(lower+1) + 'vs' + str(higher+1) + '-svm-w.txt'
            in_file_b = svm_folder + str(lower+1) + 'vs' + str(higher+1) + '-svm-b.txt'
            # import SVM and save
            w = np.loadtxt(in_file_w)
            b = np.loadtxt(in_file_b)
            w_list[i] = w
            b_list[i] = b
            
            # get confidence score for i-vs-all SVM
            c = np.dot(w,x) - b

            if c > 0:
                votes[lower] += 1
            else:
                votes[higher] += 1

            svm_results[i] = c

        # Print prediction and data
        predict = np.argmax(votes)
        most_votes = votes[predict]

        if sum(votes == most_votes) > 1:
            maxindices = np.where(votes == most_votes)[0]
            lower = maxindices[0]
            higher = maxindices[1]
            head_to_head_index = svm_pairs_reverse[str((lower + 1, higher + 1))]
            head_to_head = svm_results[head_to_head_index]
            if head_to_head > 0:
                predict = lower
            else:
                predict = higher

        if predict == scr:
            numcorrect += 1
        error += (predict - scr) * (predict - scr)

        if verbose:
            print "Predicting", k + 1, "of", num
            print ' Predicted score is %d' % (predict + 1)
            print ' Actual score is %d' % (scr + 1)
            print ' Votes:', votes
            print ' Confidence scores:', svm_results
            print ' Review text:'
            print ' ' + text[pidx]
            print

    end = time.time()

    print "Percentage Correct:", numcorrect / float(num)
    print "Mean Square Error:", error / float(num)

    print ' Time elapsed: %.4f sec per review' % ((end - start)/float(num))
