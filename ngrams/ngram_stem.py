import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from stemming.porter2 import stem
from ngrams import *

word = 'contemporaneously'
stem(word)
correct(stem(word))