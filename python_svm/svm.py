#!/usr/bin/python

import argparse
import csv
import json
import numpy as np
from constants import *
from build_svm import *
from predict import *

#########################################################################################
# Load in reviews
#########################################################################################
def load_text():
    print 'Reading in data...'
    start = time.time()

    # read in data
    f = open(review_file)
    data = f.read()
    f.close()

    # separate reviews
    sp = data.split('\n\n')
    sp = sp[:len(sp)-1]   # last one is blank

    # number of reviews
    nr = len(sp)

    # get list of different review parts
    rev_data = [sp[i].split('\n') for i in range(len(sp))]

    # initialize storage
    help_num = np.zeros(nr)
    help_den = np.zeros(nr)
    score = np.zeros(nr)
    score_ct = np.zeros(ns)
    summary = ['' for i in range(nr)]
    text = ['' for i in range(nr)]
    text_len = np.zeros(nr)
    text_wc = np.zeros(nr)

    # store review data
    for i in range(nr):
        help = rev_data[i][5][20:].split('/')
        help_num[i] = float(help[0])
        if int(help[1]) == 0:
            help_den[i] = -1.0
        else:
            help_den[i] = float(help[1])
        score[i] = float(rev_data[i][6][14:])
        score_ct[int(float(rev_data[i][6][14:]))-1] += 1
        summary[i] = rev_data[i][8][16:] 
        text[i] = rev_data[i][9][13:]
        text_len[i] = len(rev_data[i][9][13:])
        text_wc[i] = len(rev_data[i][9][13:].split())

    end = time.time()
    print ' Time elapsed: %.4f sec' % (end - start)

    return text, score


#########################################################################################
# Main function
#########################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SVM Classifier and make some number of predictions.")
    parser.add_argument("-s", "--svm", help="build the SVMs", action="store_true")
    parser.add_argument("-t", "--type", help="select type of SVMs (0 for 1-vs-all, 1 for pairwise)", type=int, default=0)
    parser.add_argument("-p","--predict", help="number of predictions to run (integer)", type=int)
    parser.add_argument("-v","--verbose", help="set flag for verbose output for predictions", action="store_true")

    args = parser.parse_args()
    
    print 'Running SVM with arguments:'
    print ' Learning set: %d' % nl
    print ' Min count: %d' % min_ct
    print ' Common words removed: %d' % ncommon
    print ' Features from each category: %d' % ntop
    print ' Penalty factor: %2f' % C
    print ' Highest n-gram: %d' % ng

    text, score = load_text()
    if not args.type:
        if args.svm:
            build_SVMs_onevsall(text, score)
        if args.predict:
            predict_onevsall(text, score, args.predict, args.verbose)
    else:
        if args.svm:
            build_SVMs_pair(text, score)
        if args.predict:
            predict_pair(text, score, args.predict, args.verbose)