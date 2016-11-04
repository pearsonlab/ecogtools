import pandas as pd

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

def merge_events_and_triggers(evt, trig, taskname=None):
	"""
	Combine event data from the task with triggers from the physiology.
	"""
	if taskname:
		trig_filt = trig.query("task == @taskname")
	else:
		trig_filt = trig.copy()
	trig_filt.reset_index(drop=True, inplace=True)
	return pd.concat([evt, trig_filt], axis=1)