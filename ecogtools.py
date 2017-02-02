from __future__ import print_function

import pandas as pd
import numpy as np
import mne
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns


"""
EXAMPLE SETUP code
patient = "patient_2002"

filepath_ecog = patient + "/" + "john_2002.edf"
filepath_behav = patient + "/" + "behavioral_data_2002/ToM_Loc_2002.json"
filepath_trig = patient + "/" + "2002_trigger_merged.csv"

phys, dat, trig = ecogtools.load_data(filepath_ecog, filepath_behav, filepath_trig)

taskname = "ToM_Loc"

event_names = ['quest_start', 'story_start', 'time_of_resp']

tmin = -1.
tmax = 5.

trig_condition = "quest_start"

## 3 options:
# Time series plotting with dataframe
type_of_plotting = "time_series_df"

event_id = {'story_start': 1, 'quest_start': 4, 'time_of_resp': 16}

try:
    ecogtools.loop_through_plots(phys, dat, trig, event_names, event_id, tmin, tmax, patient, taskname, trig_condition, type_of_plotting)
except ValueError: #this just catches the end when phys.ch_names has the strange trigger channels in it.
    print("Done")

# Time series plotting with MNE
type_of_plotting = "time_series_mne"

event_id = {'b/story_start': 1, 'b/quest_start': 4, 'b/time_of_resp': 16,
    'p/story_start': 2, 'p/quest_start': 5, 'p/time_of_resp': 17}

try:
    ecogtools.loop_through_plots(phys, dat, trig, event_names, event_id, tmin, tmax, patient,
    taskname, trig_condition, type_of_plotting)
except ValueError:
    print("Done")

# Time frequency plotting
event_id = {'b/story_start': 1, 'b/quest_start': 4, 'b/time_of_resp': 16,
    'p/story_start': 2, 'p/quest_start': 5, 'p/time_of_resp': 17}

freqs = np.arange(2, 100, 5)
n_cycles = freqs/2.

type_of_plotting = "time_frequency"

try:
    ecogtools.loop_through_plots(phys, dat, trig, event_names, event_id, tmin, tmax, patient, taskname,
    trig_condition, type_of_plotting, freqs=freqs, n_cycles=n_cycles)
except ValueError:
    print("Done")
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


def initialize_epochs_object(phys, events, event_id, channels_of_interest, tmin= -0.2, tmax=0.5):
    """
    Given ecog data phys, events, and event_id, plus option tmnin,
    tmax, and channels of interest (picks), create MNE epochs object.
    """
    channel_indices = mne.pick_channels(phys.ch_names, channels_of_interest)
    epochs_mne = mne.Epochs(phys, events, event_id=event_id, tmin=tmin, tmax=tmax, picks = channel_indices, add_eeg_ref=False)

    return epochs_mne


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


def create_mne_epochs(dat, event_names, trig, taskname, phys, event_id, channel, tmin, tmax):
    """
    Given behavioral dataframe, event_names and trigger dataframe,
    combine dataframes, set up new trigger dataframe for more native
    MNE event_ids for ToM Localizer task and initialize MNE epochs object.
    """

    evt = melt_events(dat, event_names)
    trig_merge = merge_events_and_triggers(evt, trig, taskname=taskname)

    for i in range(len(trig_merge)):
        if trig_merge.loc[i, "trial_cond"] == "p":
            trig_merge.loc[i, "trigger"] += 1

    events = define_events(trig_merge)

    epochs = initialize_epochs_object(phys, events, event_id, channel, tmin=tmin, tmax=tmax)

    return epochs


def create_evoked(epochs):
    """
    Load epochs data, split epochs object to two evoked responses
    for belief and photograph condition, combine (subtracting belief
    minus photograph) and return three evoked response objects.
    """

    epochs.load_data()

    evoked_qb = epochs['b/quest_start'].average()
    evoked_qp = epochs['p/quest_start'].average()

    evoked_qbp = mne.combine_evoked([evoked_qb, evoked_qp], weights=[1, -1])

    return evoked_qbp, evoked_qb, evoked_qp


def plot_time_series(evoked_qbp, evoked_qb, evoked_qp, channel, patient, taskname):
    """
    Using native MNE evoked (averaged epochs) objects, plot three time series plots
    (for ToM Localizer task): belief condition, photograph condition,
    and belief minus photograph condition. Save files to new directory in
    patient's data directory.
    """

    folder = patient + '/' + taskname + "_TS_mne_images" + "/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    fig = evoked_qb.plot()
    title = patient + " " + taskname + " " + channel[0] + " " + "TSb-p"
    filename =  title + ".png"
    fig.savefig(folder + filename)
    plt.close()

    fig = evoked_qp.plot()
    title = patient + " " + taskname + " " + channel[0] + " " + "TSb"
    filename =  title + ".png"
    fig.savefig(folder + filename)
    plt.close()

    fig = evoked_qbp.plot()
    title = patient + " " + taskname + " " + channel[0] + " " + "TSp"
    filename =  title + ".png"
    fig.savefig(folder + filename)
    plt.close()


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


def loop_through_plots(phys, dat, trig, event_names, event_id, tmin, tmax, patient, taskname, trig_condition, type_of_plotting, freqs=None, n_cycles=None):
    """
    Given patient data, loop through plotting and saving figures for all channels.

    type_of_plotting:
    'time_series_df' = time series plot using epochs dataframe (slower, but ~more flexible)
    'time_series_mne' = time series plot using native mne epochs objects (faster)
    'time_frequency' = time frequency plot using native mne evoked objects
    """
    for i in np.arange(len(phys.ch_names)):
        print()
        print ("{}".format(phys.ch_names[i]))
        channels_of_interest = [phys.ch_names[i]]

        if type_of_plotting == "time_series_df":
            epochs, epochs_mne = merge_to_final_epochs_df(phys, dat, trig, event_names, event_id, channels_of_interest, tmin=tmin, tmax=tmax, taskname=taskname)

            plot_dataframe(patient, epochs, taskname, phys.ch_names[i], trig_condition=trig_condition)

        elif type_of_plotting == "time_series_mne":
            epochs = create_mne_epochs(dat, event_names, trig, taskname, phys, event_id, channels_of_interest, tmin, tmax)
            evoked_qbp, evoked_qb, evoked_qp = create_evoked(epochs)

            plot_time_series(evoked_qbp, evoked_qb, evoked_qp, channels_of_interest, patient, taskname)

        elif type_of_plotting == "time_frequency":
            epochs = create_mne_epochs(dat, event_names, trig, taskname, phys, event_id, channels_of_interest, tmin, tmax)
            evoked_qbp, evoked_qb, evoked_qp = create_evoked(epochs)

            plot_tf(evoked_qbp, freqs, n_cycles, channels_of_interest, patient, taskname)
