import pandas as pd
import numpy as np
import mne
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

from __future__ import print_function

"""
EXAMPLE SETUP code

filepath_ecog = "patient_2003/john_2003.edf"
filepath_behav = "patient_2003/behavioral_data/ToM_Task_2010_2003.json"
filepath_trig = "patient_2003/2003_trigger_merged.csv"

taskname = "Tom 2010"
event_names = ['quest_start', 'story_start', 'time_of_response']
event_id = {'story_start': 1, 'quest_start': 4, 'time_of_response': 16}

channels_of_interest = ['LTG22', 'LTG29']
tmin = -0.2
tmax = 0.5

epochs, phys, dat, trig, epochs_mne = ecogtools.preprocess_data(filepath_ecog, filepath_behav, filepath_trig, event_names, event_id, channels_of_interest, taskname=taskname)
"""

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
	epochs = mne.Epochs(phys, events, event_id=event_id, tmin=tmin, tmax=tmax, picks = channel_indices, add_eeg_ref=False)
	epochs_df = epochs.to_data_frame(index='time', scale_time=10000)
	epochs_df_melt = pd.melt(epochs_df.reset_index(), 
								id_vars=['time', 'condition', 'epoch'], 
								var_name='channel', 
								value_name='voltage')


	return epochs_df_melt, epochs


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
	epochs_df_melt, epochs = initialize_epochs_dataframe(phys, events, event_id, channels_of_interest, tmin=tmin, tmax=tmax )
	ep_df = merge_epochs_df_trig_and_evt(trig_merge, epochs_df_melt)

	return ep_df, epochs
 
def preprocess_data(filepath_ecog, filepath_behav, filepath_trig, event_names, event_id, 
								channels_of_interest, tmin=-0.2, tmax=0.5, taskname=None):
	"""
	Load data and create epochs dataframe based on channels of interest,
	tmin and tmax, and task (including event_names and ids).
	"""
	
	phys, dat, trig = load_data(filepath_ecog, filepath_behav, filepath_trig)
	epochs, epochs_mne = merge_to_final_epochs_df(phys, dat, trig, event_names, event_id, channels_of_interest, 
										tmin=tmin, tmax=tmax, taskname=taskname)

	return epochs, phys, dat, trig, epochs_mne

def plot_dataframe(patient, epochs, taskname, channel_i, condition='quest_start'):
	title = patient + " " + taskname + " " + channel_i
	fig = plt.figure(figsize=(12, 9))
	plt.title(title)
	axes = plt.gca()
	axes.set_ylim([-120, 120])
    
	query_string = 'channel == "{}"  & condition == "{}"'.format(channel_i, condition) 
	sns.tsplot(epochs.query(query_string), unit='epoch', condition='trial_cond', time='time', value='voltage')
    
	folder = patient + '/' + taskname + "_images" + "/"
	filename =  title + ".png"
    
	if not os.path.exists(folder):
		os.makedirs(folder)

	fig.savefig(folder + filename)
	plt.close()

def loop_through_plots(phys, dat, trig, event_names, event_id, tmin, tmax, patient, taskname, condition):

	for i in np.arange(len(phys.ch_names)):
		print()
		print ("{}".format(phys.ch_names[i])
		channels_of_interest = [phys.ch_names[i]]
	    
		epochs, epochs_mne = merge_to_final_epochs_df(phys, dat, trig, event_names, event_id, channels_of_interest, tmin=tmin, tmax=tmax, taskname=taskname)
	    
		plot_dataframe(patient, epochs, taskname, phys.ch_names[i], condition=condition)



