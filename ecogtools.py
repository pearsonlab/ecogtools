from __future__ import print_function

import pandas as pd
import numpy as np
import mne
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

class Data:

    def __init__(self, patient_num, taskname, event_names, event_id):
        self.patient_num = patient_num
        self.taskname = taskname
        self.event_names = event_names
        self.event_id = event_id

        self.ecogfile = "patient_" + patient_num + "/" + "john_" + patient_num + ".edf"
        self.trigfile = "patient_" + patient_num + "/" + patient_num + "_trigger_merged.csv"

        if self.taskname == "ToM_Loc":
            self.behavfile = "patient_" + patient_num + "/" + "behavioral_data_" + patient_num + "/ToM_Loc_" + patient_num + ".json"

            self.load_data()
            self.tom_loc_triggers()

        elif self.taskname == "ToM_2010":
            self.behavfile = "patient_" + patient_num + "/" + "behavioral_data_" + patient_num + "/ToM_Task_2010_" + patient_num + ".json"

            self.load_data()
            self.tom_2010_triggers()

        else:
            print("You haven't set up other tasks yet.")

        self.check_directories()

        self.evoked_list = []

        
    def load_physiology_data(self):
        """
        Given filepath for ecog .edf file,
        return mne raw_data object of ecog time series data.
        """

        self.phys = mne.io.read_raw_edf(self.ecogfile, preload=False)


    def load_behavioral_data(self):
        """
        Given filepath for behavioral data json file,
        return dataframe of behavioral data to be used
        to make events dataframe.
        """

        with open(self.behavfile) as fp:
            beh = json.load(fp)

        dat_list = beh['data']
        self.dat = pd.DataFrame(dat_list)


    def load_trigger_data(self):
        """
        Given filepath for trigger csv file,
        return dataframe of trigger data.
        """
        self.trig = pd.read_csv(self.trigfile, index_col=0, dtype={'trigger':'int64', 'trigger_index':'int64'})


    def melt_events(self):
        """
        Given a dataframe dat with one line per trial and a list of
        strings giving names of columns to consider as events,
        return a dataframe with one line per (trial, event) combination.
        """
        id_names = [c for c in self.dat.columns if not c in self.event_names]
        evt = pd.melt(self.dat, id_vars=id_names, value_vars=self.event_names, var_name='trigger_name', value_name='cpu_trigger_time')
        evt.sort_values(by='cpu_trigger_time', inplace=True)
        evt.reset_index(drop=True, inplace=True)
        self.evt = evt


    def merge_events_and_triggers(self):
        """
        Combine behavioral data from the task with triggers from the csv file.
        """
        
        trig_filt = self.trig.query("task == @self.taskname")
        trig_filt.reset_index(drop=True, inplace=True)
        self.trig_and_behav =pd.concat([self.evt, trig_filt], axis=1)


    def define_events(self):
        """
        Given dataframe with trigger information from csv file,
        create numpy array of tuples that MNE uses to define events 
        in format (trigger value, 0,  trigger number)
        where trigger value corresponds to timing in physiology 
        data (edf).
        """

        self.events = np.c_[self.trig_and_behav['trigger_index'].values, np.zeros(self.trig_and_behav.shape[0], dtype='int64'), self.trig_and_behav['trigger'].values]


    def load_data(self):
        """
        Load all data (physiology-edf, triggers-csv, behavioral-json).
        Combine triggers and behavioral data to common dataframe.
        Create array for defining events in MNE.
        """
        # Load data from file
        self.load_physiology_data()
        self.load_behavioral_data()
        self.load_trigger_data()

        # Important information from edf file
        self.ch_names = self.phys.ch_names
        self.sfreq = self.phys.info['sfreq']

        # Unpivot behavioral dataframe
        self.melt_events()
        
        # Combine behavioral data and triggers.
        self.merge_events_and_triggers()

        # Create array for MNE
        self.define_events()

    def check_directories(self):
        folder = "patient_" + self.patient_num + '/' + self.taskname + "_plots/"

        if not os.path.exists(folder):
            os.makedirs(folder)


    def tom_loc_triggers(self):
        """
        Add one to trigger numbers for photograph condition
        to distinguish trigger events for MNE.
        1, 4, 16 (belief) becomes 2, 5, 17 (photograph).
        """
        for i in range(len(self.trig_and_behav)):
            if self.trig_and_behav.loc[i, "trial_cond"] == "p":
                self.events[i, 2] += 1
                self.trig_and_behav.loc[i, "trigger"] += 1

    def tom_2010_triggers(self):
        """
        COMMENT
        """
        for i in range(len(self.trig_and_behav)):
            if self.trig_and_behav.loc[i, "trigger_name"] == "quest_start":
                if self.trig_and_behav.loc[i, "state"] == "mental" and self.trig_and_behav.loc[i, "condition"] == "unexpected":
                    self.events[i, 2] += 1
                    self.trig_and_behav.loc[i, "trigger"] += 1
                elif self.trig_and_behav.loc[i, "state"] == "physical":  
                    if self.trig_and_behav.loc[i, "condition"] == "expected":
                        self.events[i, 2] += 2
                        self.trig_and_behav.loc[i, "trigger"] += 2
                    elif self.trig_and_behav.loc[i, "condition"] == "unexpected":
                        self.events[i, 2] += 3
                        self.trig_and_behav.loc[i, "trigger"] += 3


    def initialize_epochs_object(self, channels_of_interest, tmin=-1., tmax=5.0):
        """
        Given ecog data phys, events, and event_id, plus option tmnin,
        tmax, and channels of interest (picks), create MNE epochs object.
        """
        
        channel_indices = [i for i, j in enumerate(self.ch_names) for k in channels_of_interest if j == k]
        self.epochs = mne.Epochs(self.phys, self.events, event_id=self.event_id, tmin=tmin, tmax=tmax, 
                                picks = channel_indices, add_eeg_ref=False)
        self.epochs.load_data()


    def create_evoked(self, condition):
        """
        Average across epochs for one condition of task.
        Append to list of all evoked variables created.
        Return evoked object.
        """
        evoked = self.epochs[condition].average()
        self.evoked_list.append(evoked)

        return evoked

    def compute_power(self, condition, **kwargs):
        """
        Computer power and inter-trial coherence using tfr_morlet on one condition
        of main epochs variable for task.
        """
        power, itc = mne.time_frequency.tfr_morlet(self.epochs[condition], **kwargs)

        return power, itc

    def compute_diff_power(self, power1, power2):
        """
        Create AverageTFR object of ratio of two powers.
        
        """
        combined = power1 - power2

        combined.data = power1.data / power2.data

        return combined


if __name__ == "__main__":
    """
    Example use case for ecogtools
    If run through command line, must be located in
    ecog_data_analysis folder. See Jupyter Notebook
    for directories structure. Otherwise, can be imported
    as module.
    """
    # Define variables
    patient_num = "2002"

    taskname = "ToM_Loc"

    event_names = ['quest_start', 'story_start', 'time_of_resp']

    event_id = {'b/story_start': 1, 'b/quest_start': 4, 'b/time_of_resp': 16,
                'p/story_start': 2, 'p/quest_start': 5, 'p/time_of_resp': 17}

    # Initial data processing
    data = Data(patient_num, taskname, event_names, event_id)

    # Choose channel of interest
    channels_of_interest = ['RTG31']
    data.initialize_epochs_object(channels_of_interest)

    # Average data for two conditions
    evoked_belief = data.create_evoked("b/quest_start")
    evoked_photo = data.create_evoked("p/quest_start")

    # Subtract averaged data
    evoked_combined = mne.combine_evoked(data.evoked_list, weights=[1, -1])

    # Plot
    evoked_belief.plot()
    evoked_photo.plot()
    evoked_combined.plot();

    # Time frequency plot of two conditions, plot
    freqs = np.arange(2, 100, 5)
    n_cycles = freqs/2.

    power1, itc1 = data.compute_power('b/quest_start', freqs, n_cycles)
    power2, itc2 = data.compute_power('p/quest_start', freqs, n_cycles)

    power1.plot([0], baseline=(-1., 0), mode="ratio", dB=True);
    power2.plot([0], baseline=(-1., 0), mode="ratio", dB=True);

    # Find ratio of two powers, plot
    combined = data.compute_diff_power(power1, power2)
    combined.plot([0]);

    combined.plot([0], baseline=(-1., 0), mode="ratio", dB=True);
