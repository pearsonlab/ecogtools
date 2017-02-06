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
        Load all data. Return phys, ch_names, sfreq (sampling frequency),
        trig_and_behav (trigger dataframe and behavioral dataframe merged),
        and events (a numpy array that MNE epochs requires for trigger times and values).
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
        REWRITE
        """
        evoked = self.epochs[condition].average()
        self.evoked_list.append(evoked)

        return evoked

    def compute_power(self, condition, freqs, n_cycles):
        """
        COMMENT
        """
        power, itc = mne.time_frequency.tfr_morlet(self.epochs[condition], freqs=freqs, n_cycles=n_cycles,
                                                    use_fft=False, return_itc=True, average=True)

        return power, itc

def plot_tf(evoked_qbp, freqs, n_cycles, channel, patient, taskname):
    """
    Given evoked object (belief minus photo), use morlet wavelet on evoked object
    to find power. Plot power (TF plot) and evoked response. Save figure to patient
    directory.
    """

    power = mne.time_frequency.tfr_morlet(evoked_qbp, freqs=freqs, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=3, n_jobs=1)

    folder = patient + '/' + taskname + "_TF_images" + "/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    fig = power.plot([0], baseline=(-1., 0), mode="logratio", title=channel[0]+" TF Plot")

    title = patient + " " + taskname + " " + channel[0] + " " + "TF"
    filename =  title + ".png"
    fig.savefig(folder + filename)
    plt.close()

    fig = evoked_qbp.plot()

    title = patient + " " + taskname + " " + channel[0] + " " + "evoked"
    filename =  title + ".png"
    fig.savefig(folder + filename)
    plt.close()

### Really slow Dataframe stuff

def initialize_epochs_dataframe(epochs_mne):
    """
    Given epochs_mne object, create dataframe of epochs.
    """
    epochs_df = epochs_mne.to_data_frame(index='time', scale_time=10000)
    epochs_df["trig_condition"] = epochs_df["condition"]
    epochs_df.drop("condition", axis=1, inplace=True)
    epochs_df_melt = pd.melt(epochs_df.reset_index(),
                            id_vars=['time', 'trig_condition', 'epoch'],
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
    epochs_mne = initialize_epochs_object(phys, events, event_id, channels_of_interest, tmin=tmin, tmax=tmax)
    epochs_df_melt= initialize_epochs_dataframe(epochs_mne)
    ep_df = merge_epochs_df_trig_and_evt(trig_merge, epochs_df_melt)

    return ep_df, epochs_mne


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


def plot_dataframe(patient, epochs, taskname, channel_i, trig_condition='quest_start'):
    """
    Given the patient and epochs dataframe, plot time series data for
    ToM_Localizer or ToM_2010 task. Save figure to new directory in
    patient's data directory.
    """

    title = patient + " " + taskname + " " + channel_i
    fig = plt.figure(figsize=(12, 9))
    plt.title(title)
    axes = plt.gca()
    axes.set_ylim([-120, 120])

    if taskname == "ToM_Loc":
        query_string = 'channel == "{}"  & trig_condition == "{}"'.format(channel_i, trig_condition)
        sns.tsplot(epochs.query(query_string), unit='epoch', condition='trial_cond', time='time', value='voltage')
    elif taskname == "ToM_2010":
        colors = sns.color_palette("Paired", 4)

        query_string_1 = 'channel == "{}"  & trig_condition == "{}" & condition == "expected"'.format(channel_i, trig_condition)
        query_string_2 = 'channel == "{}"  & trig_condition == "{}" & condition == "unexpected"'.format(channel_i, trig_condition)

        sns.tsplot(epochs.query(query_string_1), unit='epoch', time='time', condition="state", ci=0,
        value='voltage', color= [colors[0], colors[2]])
        sns.tsplot(epochs.query(query_string_2), unit='epoch', condition='state', time='time', ci=0,
        value='voltage', color= [colors[3], colors[1]])
        plt.title(title+" (Light colors are expected)")


    folder = patient + '/' + taskname + "_TS_df_images" + "/"
    filename =  title + ".png"

    if not os.path.exists(folder):
        os.makedirs(folder)

    fig.savefig(folder + filename)
    plt.close()





