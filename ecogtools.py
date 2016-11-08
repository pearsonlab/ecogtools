import pandas as pd
import numpy as np
import mne

def load_physiology_data(filepath):
	phys = mne.io.read_raw_edf(filepath, preload=False)
	return phys 

def import_behavioral_data(filepath):
	"""
	Given filepath for behavioral data json file,
	return dataframe of behavioral data to be used
	to make events dataframe.
	"""

	with open(filepath) as fp:
		beh = json.load(fp)

	dat_list = beh['data']
	dat = pd.DataFrame(dat_list)

	return dat

def melt_events(dat, event_names):
	"""
	Given a dataframe dat with one line per trial and a list of 
	strings giving names of columns to consider as events, 
	return a dataframe with one line per (trial, event) combination.
	"""
	id_names = [c for c in dat.columns if not c in event_names]
	evt = pd.melt(dat, id_vars=id_names, value_vars=event_names, var_name='trigger_name', value_name='cpu_trigger_time')
	evt.sort_values(by='cpu_trigger_time', inplace=True)
	evt.reset_index(drop=True, inplace=True)
	return evt

def merge_events_and_triggers(evt, trig, task=None):
	"""
	Combine event data from the task with triggers from the physiology.
	"""
	if task:
		trig_filt = trig.query("task == @taskname")
	else:
		trig_filt = trig.copy()
	trig_filt.reset_index(drop=True, inplace=True)
	trig_merge =pd.concat([evt, trig_filt], axis=1)
	return trig_merge

def define_events(event_id):
	"""
	Define trigger events for use in epochs
	"""
