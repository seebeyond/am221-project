#########################################################################################
# Define constants and parameters
#########################################################################################

# number of rating categories
ns = 5

# number of reviews in learning set
nl = 1000

# minimum number of occurrences in corpus to be considered for feature
min_ct = 20

# number of most common words to remove
ncommon = 150

# number of top significant words for each category to use as features
ntop = 100

# penalty factor for SVM
C = 10

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