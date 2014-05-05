import numpy as np

#########################################################################################
# Define constants and parameters
#########################################################################################

# number of rating categories
ns = 5

# number of pairwise svms needed
num_svms = 0
for i in range(ns):
    num_svms += i

# tuples corresponding to pairwise svms
svm_pairs = np.zeros((num_svms,) , dtype=[('f0', '>i4'), ('f1', '>i4')])
svm_pairs_reverse = {}
k = 0
for i in range(ns):
    for j in range(i+1, ns):
        svm_pairs[k] = (i+1,j+1)
        k += 1

        svm_pairs_reverse[str((i+1,j+1))] = k - 1


# number of reviews in learning set
nl = 2500

# minimum number of occurrences in corpus to be considered for feature
min_ct = 8

# number of most common words to remove
ncommon = 150

# number of top significant words for each category to use as features
ntop = 1000

# penalty factor for SVM
C = 10

# highest cardinality of n-grams to include
ng = 1

# file containing review data
review_file = '../data/Software.txt'
# review_file = '/Users/Michael/Documents/am221-data/amazon/Software.txt'
# review_file = '/Users/Michael/Documents/am221-data/amazon/Kindle_Store.txt'   

# AMPL file directory name
ampl_folder = '../ampl_svm/'

# AMPL runfile
ampl_run = ampl_folder + 'svm.run'

# SVM directory
svm_folder = '../svm_data/'

dict_folder = '../dicts/'
