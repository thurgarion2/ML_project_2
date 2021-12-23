from load_data import *
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_dataset_session(data_path, session, area, start=2.5, end=0,select_amplitude=[1,2,3]):
  start = int(start*2000)
  end = int(end*2000)

  amp = load_stim_amps(data_path, session)
  times = load_TrialOnsets_All(data_path, session)
  hits = load_hit_indices(data_path, session)

  if(len(amp)!=len(times)):
    print('problem amp and times don\'t match')
    length = min(len(amp), len(times))
    times = times[:length]
    amp = amp[:length]
    hits = hits[:length]


  selected_time = times[np.isin(amp,select_amplitude)]
  selected_hits = hits[np.isin(amp, select_amplitude)]
  selected_samples = samples_time_to_samples_index(selected_time)
 
  lfp = load_lfp(data_path, session)[area]
  X = []
  shapes = set()

  cut = 0
  for sample in selected_samples:
    if(sample-start<len(lfp)):
      cut+=1
      X.append(lfp[int(sample)-start:int(sample)-end])


  return np.stack(X,axis=0), selected_hits[:cut]



def create_dataset(data_path, area, start=2.5, end=0,select_amplitude=[1,2,3]):
  X_all = []
  y_all = []

  for session in range(0,24): 
    areas = load_area(data_path, session)
    if area in areas:
      X, y = create_dataset_session(data_path, session, area, start, end, select_amplitude)
      X_all.append(X)
      y_all.append(y)

  return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)

def create_dataset_with_session(data_path, area, start=2.5, end=0,select_amplitude=[1,2,3]):
  sessions = {}

  for session in range(0,24): 
    areas = load_area(data_path, session)
    if area in areas:
      X, y = create_dataset_session(data_path, session, area, start, end, select_amplitude)
      sessions[session] = (X,y)

  return sessions
  
def cut_last_samples(sessions, nb_samples):
    return {i:(X[:,:-nb_samples], y) for i, (X,y) in sessions.items()}
  
def preprocess_dataset_with_session(data_path, area, start=2.5, select_amplitude=[2,3]):
    sessions = create_dataset_with_session(data_path, area, start=start, end=0,select_amplitude=select_amplitude)
    sessions = cut_last_samples(sessions,4)
    
    scaler = StandardScaler()
    sessions = {i:(scaler.fit_transform(X), y) for i, (X,y) in sessions.items()}
    
    return {i:balance_hit_miss(X,y) for i, (X,y) in sessions.items()}

def select_nb_points(X,y,n,seed=42):
    np.random.seed(42)
    shuffled = np.random.permutation(len(y))
    selected = shuffled[:n]
    return X[selected], y[selected]

def balance_hit_miss(X,y,seed=42):
    X_miss = X[y==0]
    y_miss = y[y==0]
    
    X_hit = X[y==1]
    y_hit = y[y==1]
    
    if  len(y_hit)>len(y_miss):
        X_hit, y_hit = select_nb_points(X_hit,y_hit,len(y_miss),seed)
    elif len(y_hit)<len(y_miss):
        X_miss, y_miss = select_nb_points(X_miss,y_miss,len(y_hit), seed)
    return np.concatenate([X_hit,X_miss],axis=0), np.concatenate([y_hit,y_miss],axis=0)
    
def preprocess_dataset(data_path, area, start=2.5, select_amplitude=[2,3], seed=42):
    sessions = preprocess_dataset_with_session(data_path, area,start=start, select_amplitude=select_amplitude)
    
    X_s = []
    y_s = []
    for i, (X,y) in sessions.items():
        X_s.append(X)
        y_s.append(y)
    return np.concatenate(X_s,axis=0), np.concatenate(y_s,axis=0)
        