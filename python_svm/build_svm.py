#!/usr/bin/python

import argparse
from collections import Counter
from constants import *
from helpers import *
import csv
import json
import numpy as np
import random
import re
import subprocess
import time
import sys
import Stemmer # PyStemmer - https://pypi.python.org/pypi/PyStemmer/1.0.1

# import os,sys,inspect
# # add parent directory to python path
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir) 
    

#########################################################################################
# Build SVM dict and select features
#########################################################################################
def feature_select(text, score, learn_idx, stemmer):
    #########################################################################################
    # Construct dictionary from learning set
    #########################################################################################

    print 'Constructing learning dictionaries...'
    start = time.time()

    ### Comment this section out after first time running for increased speed

    # initialize dictionaries for each score category
    learn_dicts = [{} for i in range(ns)]
    dict_all = {}

    # populate dictionaries from learning set
    for i in range(nl):
        idx = learn_idx[i]
        scr = int(score[idx]) - 1
        words = process_text(text[idx], stemmer)
#         # get rid of escaped characters and apostrophes
#         txt = text[idx].lower().replace("&quot;","'").replace("'","")
#         # reg-ex to split on
#         words = re.split("[\W\s\!+\.+,+\?+\"]", txt)
#         words = filter(bool, words)
#         words = stemmer.stemWords(words) + ngrams(words, 2)
        for word in words:
          # add to category dictionary
            if word in learn_dicts[scr].keys():
                learn_dicts[scr][word] += 1 
            else:
                learn_dicts[scr][word] = 1
            # add to global dictionary
            if word in dict_all.keys():
                dict_all[word] += 1 
            else:
                dict_all[word] = 1

    # save dictionaries so for faster processing
    for i in range(ns):
        f = open(dict_folder + "learn_dict" + str(i + 1) + ".csv", "w")
        w = csv.writer(f)
        for key, val in learn_dicts[i].items():
            w.writerow([key, val])
        f.close()
    # global dictionary
    f = open(dict_folder + "dict_all.csv", "w")
    w = csv.writer(f)
    for key, val in dict_all.items():
        w.writerow([key, val])
    f.close()

    ### End commented out section

    # read the dictionaries in
    learn_dicts = [{} for i in range(ns)]
    dict_all = {}
    for i in range(ns):
        f = open(dict_folder + "learn_dict" + str(i + 1) + ".csv", "r")
        for key, val in csv.reader(f):
            learn_dicts[i][key] = int(val)
        f.close()
    # global dictionary
    f = open(dict_folder + "dict_all.csv", "r")
    for key, val in csv.reader(f):
        dict_all[key] = int(val)
    f.close()

    # read in most common words
    # see http://norvig.com/ngrams/
    f = open("../data/count_1w.txt")
    dict_common = {}
    tmp = 0
    for l in f:
        if tmp >= ncommon:
            break
        entry = l.split()
        dict_common[entry[0]] = 1
        tmp += 1
    f.close()

    end = time.time()
    print ' Time elapsed: %.4f sec' % (end - start)


    #########################################################################################
    # Select features to use
    #########################################################################################

    print 'Selecting features...'
    start = time.time()

    # Calculate z-scores for each feature
    # see http://www.asso-aria.org/coria/2012/273.pdf for details
    score_dicts = [{} for i in range(ns)]
    for i in range(ns):
        Cm = dict(Counter(dict_all) - Counter(learn_dicts[i]))
        aplusc = sum(learn_dicts[i].values())
        bplusd = sum(Cm.values())
        n = aplusc + bplusd
        for key, val in learn_dicts[i].items():
            f = key
            # throw out if total number of occurrences is less than min_ct
            if dict_all[f] <= min_ct or f in dict_common.keys():
                continue
            a = val     # a = learn_dicts[i][f]
            if f in Cm.keys():
                b = Cm[f]
            else:
                b = 0
            Pf = (a+b)/float(n)
            Zf = (a - aplusc * Pf)/np.sqrt(aplusc * Pf * (1-Pf))
            score_dicts[i][f] = Zf

    # get words with highest z-score in each category
    sd = [[] for i in range(ns)]
    for i in range(ns):
        sd[i] = sorted(score_dicts[i].items(), key=lambda x: -x[1])

    # build the svm dictionary
    svm_dict = {}
    idx = 0
    for i in range(ns):
        end_idx = ntop if ntop < len(sd[i]) else len(sd[i])   # ensure no index error
        for k, v in sd[i][0:end_idx]:
            if k not in svm_dict.keys():
                svm_dict[k] = idx
                idx += 1
    # number of features
    nf = idx

    end = time.time()
    print ' Time elapsed: %.4f sec' % (end - start)
    
    return svm_dict, nf



#########################################################################################
# Build SVMs for 1-vs-all strategy
#########################################################################################
def build_SVMs_onevsall(text, score):
    # randomly select learning set
    nr = len(text)
    random.seed(0)
    learn_idx = random.sample(range(nr), nl)

    # initialize stemmer
    stemmer = Stemmer.Stemmer('english')

    # select features to use
    svm_dict, nf = feature_select(text, score, learn_idx, stemmer)
 
    #########################################################################################
    # Build the feature vectors for one-vs-all approach
    #########################################################################################

    print 'Building feature vectors...'
    start = time.time()

    # lists for various categories
    X_list = [np.zeros((nl,nf)) for i in range(ns)]
    Y_list = [np.zeros((nl,1)) for i in range(ns)]

    # loop over categories
    for cat in range(ns):
        # initialize objects
        X = np.zeros((nl,nf))
        Y = np.zeros((nl,1))
        row = 0

        # loop over learning reviews
        # CAN BREAK OUT X SINCE THEY ARE ALL THE SAME IN 1-VS-ALL
        for i in range(nl):
            idx = learn_idx[i]
            scr = int(score[idx]) - 1
            words = process_text(text[idx], stemmer)
            for word in words:
#                 word = correct(word)     # extract root of word
                if word in svm_dict.keys():
                    X[row, svm_dict[word]] += 1
            Y[row] = 1 if scr == cat else -1
            row += 1
        
        # save objects
        X_list[cat] = X
        Y_list[cat] = Y

    end = time.time()
    print ' Time elapsed: %.4f sec' % (end - start)


    #########################################################################################
    # Write AMPL data file and solve SVM optimization problem using AMPL
    #########################################################################################

    print 'Writing and solving AMPL data files...'

    print ' Building X matrix string...'
    start = time.time()
    
    # construct X parameter once - all X's are same for 1-vs-all
    X = X_list[0]
    # X parameter string
    sx = 'param X :'
    # initial line
    for i in range(nf):
        sx = sx + ' ' + str(i + 1)
    # end of first line
    sx = sx + ' :=\n'
    # complicated ampl matrix formatting
    np.set_printoptions(threshold=sys.maxint)
    strmat = np.transpose(np.vstack([np.arange(1,nl+1), np.transpose(X.astype(int))]))
    sxmat = np.array_str(strmat, max_line_width=sys.maxint).replace('\n','').replace('[','').replace(']','\n').replace('  ',' ')
    
    end = time.time()
    print ' Time elapsed: %.4f sec' % (end - start)

    # loop over categories
    for cat in range(ns):
        print ' Building %d star-vs-all SVM...' % (cat+1)
        start = time.time()
        
        # get data objects
#         X = X_list[cat]
        Y = Y_list[cat]
        
        # write AMPL runfile
        f = open(ampl_run, 'w')
        f.write('model ' + ampl_folder + 'svm.mod;\n')
        f.write('data ' + ampl_folder + 'svm_data' + str(cat+1) + '.dat;\n')
        f.write("option solver 'cplex';\n")
        f.write("option solver_msg 0;\n")
        f.write("solve;\n");
        f.write("option display_1col 100000000;\n")
        f.write("display w > " + ampl_folder + "svm-w.txt;\n")
        f.write("display b > " + ampl_folder + "svm-b.txt;\n")
        f.close()
        
        # open data file
        filename = ampl_folder + 'svm_data' + str(cat+1) + '.dat'
        f = open(filename,'w')
        
        # Y parameter string
        sy = ''
        for i in range(nl):
            sy = sy + ' ' + str(i+1) + ' ' + np.array_str(Y[i,0])
        
        # write lines
        f.write('data;\n')
        f.write('param n := ' + str(nl) + ';\n')
        f.write('param d := ' + str(nf) + ';\n')
        f.write('param C := ' + str(C) + ';\n')
        f.write('param Y := ' + sy + ';\n')
        f.write(sx)
        f.write(sxmat)
        f.write(';\n')
        
        # close data file
        f.close()
        
        # run ampl
        devnull = open('/dev/null', 'w')
        _ = subprocess.call(["ampl", ampl_run], stdout=devnull, stderr=devnull)
        devnull.close()
        
        # read in svm data from AMPL output
        w = np.zeros(nf)
        b = np.zeros(1)
        # get w values
        f = open(ampl_folder + "svm-w.txt")
        _ = f.readline()    # dump first line
        idx = 0
        for l in f:
            tmp = l.replace('\n','').split(' ')
            tmp = filter(bool, tmp)
            if idx < nf:
                w[idx] = float(tmp[1])
            idx += 1
        f.close()
        # get b value
        f = open(ampl_folder + "svm-b.txt")
        tmp = f.readline()
        tmp = tmp.replace('\n','').replace('=', '').split(' ')
        tmp = filter(bool, tmp)
        b[0] = float(tmp[1])
        f.close()
        
        # save svm
        out_file_w = svm_folder + str(cat+1) + '-svm-w.txt'
        out_file_b = svm_folder + str(cat+1) + '-svm-b.txt'
        np.savetxt(out_file_w, w)
        np.savetxt(out_file_b, b)
        
        end = time.time()
        print ' Time elapsed: %.4f sec' % (end - start)
    # end loop

    # save svm dictionary
    out_file_idx = svm_folder + 'svm-dict-onevsall.csv'
    f = open(out_file_idx, "w")
    w = csv.writer(f)
    for key, val in svm_dict.items():
        w.writerow([key, val])
    f.close()



#########################################################################################
# Build SVMs for pairwise strategy
#########################################################################################
def build_SVMs_pair(text, score):
    # randomly select learning set
    nr = len(text)
    random.seed(0)
    learn_idx = random.sample(range(nr), nl)

    # initialize stemmer
    stemmer = Stemmer.Stemmer('english')

    # select features to use
    svm_dict, nf = feature_select(text, score, learn_idx, stemmer)

    #########################################################################################
    # Build the feature vectors for pairwise approach
    #########################################################################################

    print 'Building feature vectors...'
    start = time.time()

    nl_array = np.zeros(ns)
    for i in range(ns):
        nl_array[i] = sum(score[learn_idx] == i + 1)

    # lists for various categories
    X_list = [None for i in range(num_svms)]
    Y_list = [None for i in range(num_svms)]

    # loop over categories
    for n in range(num_svms):
        # initialize objects
        lower = svm_pairs[n][0] - 1
        higher = svm_pairs[n][1] - 1

        X = np.zeros((nl_array[lower] + nl_array[higher],nf))
        Y = np.zeros((nl_array[lower] + nl_array[higher],1))
        row = 0

        # loop over learning reviews
        # CAN BREAK OUT X SINCE THEY ARE ALL THE SAME IN 1-VS-ALL
        for i in range(nl):
            idx = learn_idx[i]
            scr = int(score[idx]) - 1
            if scr == lower or scr == higher:
                words = process_text(text[idx], stemmer)             
                for word in words:
                    if word in svm_dict.keys():
                        X[row, svm_dict[word]] += 1
                Y[row] = 1 if scr == lower  else -1
                row += 1
        
        # save objects
        X_list[n] = X
        Y_list[n] = Y

    end = time.time()
    print ' Time elapsed: %.4f sec' % (end - start)


    #########################################################################################
    # Write AMPL data file and solve SVM optimization problem using AMPL
    #########################################################################################

    print 'Writing and solving AMPL data files...'

    # loop over categories
    np.set_printoptions(threshold=sys.maxint)    
    for n in range(num_svms):
        
        # construct X matrix and string for each SVM
        print ' Building %dth X matrix string...' % n
        start = time.time()
        X = X_list[n]

        lower = svm_pairs[n][0] - 1
        higher = svm_pairs[n][1] - 1
        npair = int(nl_array[lower]+nl_array[higher])
        
        # X parameter string
        sx = 'param X :'
        # initial line
        for i in range(nf):
            sx = sx + ' ' + str(i + 1)
        # end of first line
        sx = sx + ' :=\n'
        # complicated ampl matrix formatting
        strmat = np.transpose(np.vstack([np.arange(1,nl_array[lower] + nl_array[higher] + 1), np.transpose(X.astype(int))]))
        sxmat = np.array_str(strmat, max_line_width=sys.maxint).replace('\n','').replace('[','').replace(']','\n').replace('  ',' ')

        end = time.time()
        print ' Time elapsed: %.4f sec' % (end - start)
        
        print ' Building %d vs %d star SVM...' % (lower, higher)
        start = time.time()
        
        # get data objects
#         X = X_list[cat]
        Y = Y_list[n]
        
        # write AMPL runfile
        f = open(ampl_run, 'w')
        f.write('model ' + ampl_folder + 'svm.mod;\n')
        f.write('data ' + ampl_folder + 'svm_data' + str(lower+1) + 'vs' + str(higher + 1) + '.dat;\n')
        f.write("option solver 'cplex';\n")
        f.write("option solver_msg 0;\n")
        f.write("solve;\n");
        f.write("option display_1col 100000000;\n")
        f.write("display w > " + ampl_folder + "svm-w.txt;\n")
        f.write("display b > " + ampl_folder + "svm-b.txt;\n")
        f.close()
        
        # open data file
        filename = ampl_folder + 'svm_data' + str(lower+1) + 'vs' + str(higher + 1) + '.dat'
        f = open(filename,'w')
        
        # Y parameter string
        sy = ''
        for i in range(npair):
            sy = sy + ' ' + str(i+1) + ' ' + np.array_str(Y[i,0])
        
        # write lines
        f.write('data;\n')
        f.write('param n := ' + str(nl_array[lower] + nl_array[higher]) + ';\n')
        f.write('param d := ' + str(nf) + ';\n')
        f.write('param C := ' + str(C) + ';\n')
        f.write('param Y := ' + sy + ';\n')
        f.write(sx)
        f.write(sxmat)
        f.write(';\n')
        
        # close data file
        f.close()
        
        # run ampl
        devnull = open('/dev/null', 'w')
        _ = subprocess.call(["ampl", ampl_run], stdout=devnull, stderr=devnull)
        devnull.close()
        
        # read in svm data from AMPL output
        w = np.zeros(nf)
        b = np.zeros(1)
        # get w values
        f = open(ampl_folder + "svm-w.txt")
        _ = f.readline()    # dump first line
        idx = 0
        for l in f:
            tmp = l.replace('\n','').split(' ')
            tmp = filter(bool, tmp)
            if idx < nf:
                w[idx] = float(tmp[1])
            idx += 1
        f.close()
        # get b value
        f = open(ampl_folder + "svm-b.txt")
        tmp = f.readline()
        tmp = tmp.replace('\n','').replace('=', '').split(' ')
        tmp = filter(bool, tmp)
        b[0] = float(tmp[1])
        f.close()
        
        # save svm
        out_file_w = svm_folder + str(lower+1) + 'vs' + str(higher+1) + '-svm-w.txt'
        out_file_b = svm_folder + str(lower+1) + 'vs' + str(higher+1) + '-svm-b.txt'
        np.savetxt(out_file_w, w)
        np.savetxt(out_file_b, b)
        
        end = time.time()
        print ' Time elapsed: %.4f sec' % (end - start)
    # end loop

    # save svm dictionary
    out_file_idx = svm_folder + 'svm-dict-pair.csv'
    f = open(out_file_idx, "w")
    w = csv.writer(f)
    for key, val in svm_dict.items():
        w.writerow([key, val])
    f.close()