param n;  # number of observations
param d;  # number of features
param C;  # penalty factor

set Obs := 1..n;
set Feats := 1..d;

param Y {Obs};
param X {Obs,Feats};

var wp {Feats} >= 0;
var wm {Feats} >= 0;
var w {Feats};
var xi {Obs} >= 0;
var bp >= 0;
var bm >= 0;
var b;

minimize obj: (sum {i in Feats} (wp[i] + wm[i])) + C * (sum {i in Obs} xi[i]);

subject to classification {i in Obs}:
    Y[i] * ( sum {j in Feats} (wp[j] - wm[j])*X[i,j] - (bp - bm)) >= 1 - xi[i];
  
subject to abs_val {i in Feats}:
    w[i] = wp[i] - wm[i];

subject to b_val:
    b = bp - bm;