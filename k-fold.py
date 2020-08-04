"""
Script to divide in K folds for cross validation

Machine Learning Project

"""
import pickle
import numpy as np

d = list(pickle.load(open('../data/nci_features.pickle', 'rb')))
keys = d[0].keys()

conc_d = d[0]

#Concatenate previous train, validation and test sets
for k in keys:
    conc_d[k] = np.concatenate((d[0][k],d[1][k],d[2][k]), axis=0)
    
#Slip them randomly to create new subsets
for i in range(0,5):
    print(i)        
    # partitioning 70-15-15 
    # partition considering patients' numbers
    n = conc_d['patient'].max()+1 
    ix = np.random.choice(n, n, False)
    ixs = np.split(ix, [int(n*0.70), int(n*(0.70+0.15))])
    
    res_d = ()
    for ix in ixs:
        res_d += ({k: np.concatenate([v[conc_d['patient'] == i] for i in ix]) for k, v in conc_d.items()},)
    
    pickle.dump(res_d, open('nci_features_%d.pickle' % i, 'wb'))
    