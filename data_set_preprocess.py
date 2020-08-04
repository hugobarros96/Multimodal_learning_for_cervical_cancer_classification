from tensorflow.keras import utils
from skimage import draw
import pickle
import numpy as np

def bbox_to_map(x1, y1, x2, y2):
    bbmap = np.zeros((224, 224, 1), np.float32)
    start = (int(y1*224), int(x1*224))
    end = (int(y2*224), int(x2*224))
    rr, cc = draw.rectangle(start, end=end, shape=(224, 224))
    bbmap[rr, cc] = 1
    return bbmap


def load_nci_bin(ds_number):
    # binary classification; Only considers 2 classes
    d = list(pickle.load(open('data/nci_features_%d.pickle' % ds_number, 'rb')))
    # d = list(pickle.load(open('data/nci_features.pickle', 'rb')))
    # aproveitar teste do HPV para inferir o resultado do Hist
    # converter nan values of ['wrst_hist_after_dt'] in numeric values 
    for i, part in enumerate(d):
        ix = np.logical_and(
            part['wrst_hist_after'] == -2,
            np.logical_or(part['hpv_status'] == 0, part['hpv_status'] == 1))
        part['wrst_hist_after'][ix] = 0
        
        #substituição de valores nulos no wrst_hist_after_dt
        ix = np.isnan(part['wrst_hist_after_dt'])
        part['wrst_hist_after_dt'][ix] = -10
        
        ix = part['wrst_hist_after'] >= 0
        d[i] = {k: v[ix] for k, v in part.items()}
        
    # passar de 5 classes para 2
    for i, part in enumerate(d):
        ix_zero = np.logical_or(part['wrst_hist_after'] == 0,
                                 part['wrst_hist_after'] == 1)
        part['wrst_hist_after'][ix_zero] = 0
        
        ix_one = np.logical_or(part['wrst_hist_after'] == 2,
                               np.logical_or(part['wrst_hist_after'] == 3,
                                             part['wrst_hist_after'] == 4))
        part['wrst_hist_after'][ix_one] = 1

    Xtr = (d[0]['image'] / 255).astype(np.float32)
    Xval = (d[1]['image'] / 255).astype(np.float32)
    Xts = (d[2]['image'] / 255).astype(np.float32)

    Ytr = utils.to_categorical(d[0]['wrst_hist_after'])
    Yval = utils.to_categorical(d[1]['wrst_hist_after'])
    Yts = utils.to_categorical(d[2]['wrst_hist_after'])

    BBtr = np.array([bbox_to_map(*bb) for bb in d[0]['bbox']])
    BBval = np.array([bbox_to_map(*bb) for bb in d[1]['bbox']])
    BBts = np.array([bbox_to_map(*bb) for bb in d[2]['bbox']])
    
    PHOGtr = (d[0]['phog']).astype(np.float32)
    PHOGval = (d[1]['phog']).astype(np.float32)
    PHOGts = (d[2]['phog']).astype(np.float32)
    
    PLABtr = (d[0]['plab']).astype(np.float32)
    PLABval = (d[1]['plab']).astype(np.float32)
    PLABts = (d[2]['plab']).astype(np.float32)
    
    PLBPtr = (d[0]['plbp']).astype(np.float32)
    PLBPval = (d[1]['plbp']).astype(np.float32)
    PLBPts = (d[2]['plbp']).astype(np.float32)
    
    Feattr = np.concatenate((PHOGtr, PLABtr, PLBPtr), axis=-1)
    Featval = np.concatenate((PHOGval, PLABval, PLBPval), axis=-1)
    Featts = np.concatenate((PHOGts, PLABts, PLBPts), axis=-1)
    
    agetr = d[0]['age_grp'][:, np.newaxis] / np.amax(d[0]['age_grp'][:, np.newaxis])
    ageval = d[1]['age_grp'][:, np.newaxis] / np.amax(d[0]['age_grp'][:, np.newaxis])
    agets = d[2]['age_grp'][:, np.newaxis] / np.amax(d[0]['age_grp'][:, np.newaxis])
   
    hpvtr = d[0]['hpv_status'][:, np.newaxis] / np.amax(d[0]['hpv_status'][:, np.newaxis])
    hpvval = d[1]['hpv_status'][:, np.newaxis] / np.amax(d[0]['hpv_status'][:, np.newaxis])
    hpvts = d[2]['hpv_status'][:, np.newaxis] / np.amax(d[0]['hpv_status'][:, np.newaxis])
    
    timetr = d[0]['timepnt'][:, np.newaxis] / np.amax(d[0]['timepnt'][:, np.newaxis])
    timeval = d[1]['timepnt'][:, np.newaxis] / np.amax(d[0]['timepnt'][:, np.newaxis])
    timets = d[2]['timepnt'][:, np.newaxis] / np.amax(d[0]['timepnt'][:, np.newaxis])
    
    ydttr = d[0]['wrst_hist_after_dt'][:, np.newaxis] / np.amax(d[0]['wrst_hist_after_dt'][:, np.newaxis])
    ydtval = d[1]['wrst_hist_after_dt'][:, np.newaxis] / np.amax(d[0]['wrst_hist_after_dt'][:, np.newaxis])
    ydtts = d[2]['wrst_hist_after_dt'][:, np.newaxis] / np.amax(d[0]['wrst_hist_after_dt'][:, np.newaxis])
    
    Clintr = np.concatenate((agetr, hpvtr, timetr, ydttr), axis=-1)
    Clinval = np.concatenate((ageval, hpvval, timeval, ydtval), axis=-1)
    Clints = np.concatenate((agets, hpvts, timets, ydtts), axis=-1)

    return (Xtr, Xval, Xts), (Ytr, Yval, Yts), (BBtr, BBval, BBts), (Feattr, Featval, Featts), (Clintr, Clinval, Clints)

def load_kaggle():
    d = pickle.load(open('data/kaggle.pickle', 'rb'))

    Xtr = (d[0]['image'] / 255).astype(np.float32)
    Xval = (d[1]['image'] / 255).astype(np.float32)
    Xts = (d[2]['image'] / 255).astype(np.float32)

    Ytr = utils.to_categorical(d[0]['type'])
    Yval = utils.to_categorical(d[1]['type'])
    Yts = utils.to_categorical(d[2]['type'])

    return (Xtr, Xval, Xts), (Ytr, Yval, Yts)