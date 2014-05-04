import numpy as np
import time


nf = 300
nl = 300
X = np.zeros((nl,nf))

start = time.time()

# X parameter string
sx = 'param X :'
# initial line
for i in range(nf):
    sx = sx + ' ' + str(i + 1)
# end of first line
sx = sx + ' :=\n'
# create matrix body
for i in range(nl):
    sx = sx + '          ' + str(i + 1)
    for j in range(nf):
        sx = sx + ' ' + np.array_str(X[i,j])
    sx = sx + '\n'
# take off last break and add in semicolon
sx = sx[:-1] + ';\n'

end = time.time()
print 'Loop Method: %.4f' % (end - start)

# print sx

start = time.time()

sx = 'param X :'
# initial line
for i in range(nf):
    sx = sx + ' ' + str(i + 1)

# end of first line
sx = sx + ' :=\n'

# tmp = ['\n         ' + str(i+1) for i in range(nl)]
# tmp.extend([';\n',''])
# repvec = tuple(tmp)
strmat = np.transpose(np.vstack([np.arange(1,nl+1), np.transpose(X.astype(int))]))
sx = sx + np.array_str(strmat).replace('\n','').replace('[','').replace(']','\n')
# sx = sx + np.array_str(X).replace('[','').replace(']','').replace('  ',' ')


# for i in range(nl):
#     sx = sx + '\n         ' + str(i+1) + 
#     sx.replace('\n', '\n         ' + str(i + 1))

sx = sx[:-4] + ';\n'

# sx = sx % repvec

end = time.time()
print 'Replace Method: %.4f' % (end - start)

# print sx
