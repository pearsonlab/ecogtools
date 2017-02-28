from __future__ import print_function

import pandas as pd
import numpy as np
import mne
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns

class Data:

    def __init__(self):
        for key, value in self.parameters.items():
            setattr(self, key, value)

        self.load_data()

        self.check_directories()

        self.evoked_list = []

        
    def load_physiology_data(self):
        """
        Given filepath for ecog file,
        return mne raw_data object of ecog time series data.
        """

        if self.ecogfile.endswith(".edf"):
            self.phys = mne.io.read_raw_edf(self.ecogfile, preload=False)
        elif self.ecogfile.endswith(".fif"):
            self.phys = mne.io.read_raw_fif(self.ecogfile, preload=False)


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

        self.image_folder = folder


    def remove_irrelevant_channels(self, *args):
        """
        Removes irrelevant channels from ch_names list for easier plotting
        (channels with no ECoG data). Can pass list of channels or use
        default list that includes Event, Stim, and two EKG channels.
        In both cases, will remove extra C### channels.
        """

        if args:
            irrelevant_channels = args
        else:
            irrelevant_channels = ["Event", "STI 014", "EKGL", "EKGR"]

        self.ch_names = [value for value in self.ch_names if not value in irrelevant_channels and not value.startswith("C")]


    def initialize_epochs_object(self, channels_of_interest, **kwargs):
        """
        Given ecog data phys, events, and event_id, plus option tmnin,
        tmax, and channels of interest (picks), create MNE epochs object.
        """
        if "tmin" not in kwargs:
            kwargs["tmin"] = -1.0
        if "tmax" not in kwargs:
            kwargs["tmax"] = 5.0

        channel_indices = [i for i, j in enumerate(self.phys.ch_names) for k in channels_of_interest if j == k]
        self.epochs = mne.Epochs(self.phys, self.events, event_id=self.event_id, picks = channel_indices, add_eeg_ref=False, **kwargs)
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



class ToM_Localizer(Data):

    def __init__(self, patient_num):
        """
        Class to import and analyze data for ToM Localizer task.
        """
        self.patient_num = patient_num
        self.taskname = "ToM_Loc"

        self.import_parameters()
        Data.__init__(self)

        self.set_triggers()

    def import_parameters(self):
        """
        Parameters are .json files saved in
        ecog_data_analysis for specific patients
        and specific tasks. They are imported and
        used as attributes that are passed to class.
        """
        filepath = self.patient_num + "_analysis.json"
        with open(filepath) as fp:
            parameters = json.load(fp)

        with open("ToM_Loc_analysis.json") as fp:
            parameters_task = json.load(fp)

        parameters_task["behavfile"] = parameters["behavfilefolder"]+parameters_task["behavfile"]+parameters["patient_num"]+".json"
        parameters_task.update(parameters)

        self.parameters = parameters_task


    def set_triggers(self):
        """
        Add one to trigger numbers for photograph condition
        to distinguish trigger events for MNE.
        1, 4, 16 (belief) becomes 2, 5, 17 (photograph).
        """
        for i in range(len(self.trig_and_behav)):
            if self.trig_and_behav.loc[i, "trial_cond"] == "p":
                self.events[i, 2] += 1
                self.trig_and_behav.loc[i, "trigger"] += 1



class ToM_2010(Data):

    def __init__(self, patient_num):
        """
        Class to import and analyze data for ToM 2010 task.
        """
        self.patient_num = patient_num
        self.taskname = "ToM_2010"

        self.import_parameters()
        Data.__init__(self)

        self.set_triggers()


    def import_parameters(self):
        """
        Parameters are .json files saved in
        ecog_data_analysis for specific patients
        and specific tasks. They are imported and
        used as attributes that are passed to class.
        """
        filepath = self.patient_num + "_analysis.json"
        with open(filepath) as fp:
            parameters = json.load(fp)

        with open("ToM_2010_analysis.json") as fp:
            parameters_task = json.load(fp)

        parameters_task["behavfile"] = parameters["behavfilefolder"]+parameters_task["behavfile"]+parameters["patient_num"]+".json"
        parameters_task.update(parameters)

        self.parameters = parameters_task


    def set_triggers(self):
        """
        Add one to trigger numbers to distinguish between
        four distinct cominations of events (all within quest_start
        for now): mental x physical & expected x unexpected.
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
