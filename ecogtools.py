import pandas as pd
import numpy as np
import mne
import json

def load_physiology_data(filepath):
	"""
	Given filepath for ecog .edf file,
	return mne raw_data object of ecog time series data.
	"""

	phys = mne.io.read_raw_edf(filepath, preload=False)
	return phys 


def load_behavioral_data(filepath):
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


def load_trigger_data(filepath):
	"""
	Given filepath for trigger csv file,
	return dataframe of trigger data.
	"""
	trig = pd.read_csv(filepath, index_col=0, dtype={'trigger':'int64', 'trigger_index':'int64'})
	return trig


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
	trig_merge =pd.concat([evt, trig_filt], axis=1)
	return trig_merge


def define_events(trig):
	"""
	Given trig, define trigger events for use in epochs.
	"""

	events = np.c_[trig['trigger_index'].values, np.zeros(trig.shape[0], dtype='int64'), trig['trigger'].values]

	return events


def initialize_epochs_dataframe(phys, events, event_id, channels_of_interest, tmin= -0.2, tmax=0.5):
	"""
	Given ecog data phys, events, and event_id, plus option tmnin,
	tmax, and channels of interest (picks), create epoch object.
	Then 
	"""
	channel_indices = mne.pick_channels(phys.ch_names, channels_of_interest)
	epochs = mne.Epochs(phys, events, event_id=event_id, tmin=tmin, tmax=tmax, picks = channel_indices)
	epochs_df = epochs.to_data_frame(index='time', scale_time=10000)
	epochs_df_melt = pd.melt(epochs_df.reset_index(), 
								id_vars=['time', 'condition', 'epoch'], 
								var_name='channel', 
								value_name='voltage')


	return epochs_df_melt


def merge_epochs_df_trig_and_evt(trig_merge, epochs_df_melt):
	"""
	Given merged trigger and evt dataframes (trig_merge) and
	epochs df, merge.
	"""

	ep_df = epochs_df_melt.merge(trig_merge, left_on='epoch', right_index=True)

	return ep_df

def load_data(filepath_ecog, filepath_behav, filepath_trig):
	"""
	Load all data
	"""
	phys = load_physiology_data(filepath_ecog)
	dat = load_behavioral_data(filepath_behav)
	trig = load_trigger_data(filepath_trig)

	return phys, dat, trig


def merge_to_final_epochs_df(phys, dat, trig, event_names, event_id, 
							channels_of_interest, tmin=-0.2, tmax=0.5, taskname=None):
	"""
	Take all loaded data from phys, dat, and trig,
	merge to create final epochs dataframe using 
	functions to merge events and triggers, define events,
	and initialize epochs object.
	"""
	evt = melt_events(dat, event_names)
	trig_merge = merge_events_and_triggers(evt, trig, taskname=taskname)
	events = define_events(trig)
	epochs_df_melt = initialize_epochs_dataframe(phys, events, event_id, channels_of_interest, tmin=tmin, tmax=tmax )
	ep_df = merge_epochs_df_trig_and_evt(trig_merge, epochs_df_melt)

	return ep_df
 
def preprocess_data(filepath_ecog, filepath_behav, filepath_trig, event_names, event_id, 
								channels_of_interest, tmin=-0.2, tmax=0.5, taskname=None):
	"""
	Load data and create epochs dataframe based on channels of interest,
	tmin and tmax, and task (including event_names and ids).
	"""
	
	phys, dat, trig = load_data(filepath_ecog, filepath_behav, filepath_trig)
	epochs = merge_to_final_epochs_df(phys, dat, trig, event_names, event_id, channels_of_interest, 
										tmin=tmin, tmax=tmax, taskname=taskname)

	return epochs






