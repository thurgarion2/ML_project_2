import h5py
import numpy as np
import pandas as pd
 
def load_area(data_path, session):
  ref = selected_field(data_path, 'Area', session)
  areas_ref = flatten_dataset(load_ref(data_path,ref))
  return [load_text(data_path, ref) for ref in  areas_ref]
 
def load_hit_indices(data_path, session):
  return load_indices(data_path, session, 'HitIndices')
 
def load_miss_indices(data_path, session):
  return load_indices(data_path, session, 'MissIndices')
 
def load_stim_indices(data_path, session):
  return load_indices(data_path, session, 'StimIndices')
 
def load_stim_amps(data_path, session):
  return load_indices(data_path, session, 'StimAmps')
 
def load_TrialOnsets_All(data_path, session):
  ref = selected_field(data_path, 'TrialOnsets_All', session)
  return np.array(load_ref(data_path,ref)).squeeze()
 
def load_dates(data_path, session):
  ref = selected_field(data_path, 'date', session)
  return load_text(data_path, ref)
 
def nb_stimulation(data_path, session):
  return len(load_stim_indices(data_path, session))
 
def load_lfp(data_path, session):
  ref = selected_field(data_path, 'LFP', session)
  lfp_ref = flatten_dataset(load_ref(data_path,ref))
  labels = load_area(data_path, session)
 
  return {label :np.array(load_ref(data_path,ref)).squeeze() for label, ref in zip(labels,lfp_ref)}
 
 
def lfp_folder(data_path):
  return h5py.File(data_path)['DataLFP']
 
def load_ref(data_path,ref):
  return lfp_folder(data_path)[ref]
 
def load_text(data_path, ref):
  return bytes_to_string(flatten_dataset(load_ref(data_path, ref)))
 
def bytes_to_string(bytes_):
  return ''.join([chr(c) for c in bytes_])
 
def load_indices(data_path, session, field):
  ref = selected_field(data_path, field, session)
  return np.array(load_ref(data_path,ref)[0])
 
 
### session between 0 and 23
def fields(data_path):
  return lfp_folder(data_path).keys()
 
 
def flatten_dataset(ds):
  ds = [x for y in ds for x in y]
  return ds 
 
def selected_field(data_path, field, session):
  return flatten_dataset(lfp_folder(data_path)[field])[session]
 
def samples_time_to_samples_index(samples_time):
  sampling_rate = 2000
  return np.round(sampling_rate*samples_time)
