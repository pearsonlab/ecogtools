# ecogtools
functions for ecog data analysis

**See example notebooks in [pearsonlab/ecog_data_analysis_notebooks](https://github.com/pearsonlab/ecog_data_analysis_notebooks) for best examples of analysis workflow.**

**See comments in [ecogtools.py](https://github.com/pearsonlab/ecogtools/blob/master/ecogtools.py) file for specifics about different functions/class.**

# Use
Class Data has the majority of the functions for analysis, while subclasses (e.g. ToM_Localizer) have task-specific parameters.

1. Calling e.g. ToM_Localizer class
    - accepts "patient_num" : an integer value assigned to each patient (usually in #2000s)
    - searches for analysis.json files with parameters for:
      - Patient
      - Task
    - Parameter files contain directory information for patients, event information and locations of relevant files.
2. Does following automatically:
    - Loads physiology data (ecog- fif/edf file)
    - Loads behavioral data (task - json file)
    - Loads trigger data (task/ecog - csv file)
    - Combines behavioral data and triggers
    - Defines events for MNE
    - Checks directories (for saving image files)
    - Sets triggers (task/class-specific) to differentiate conditions
3. Additional functions within Data Class:
    - **Remove_irrelevant_channels** – helpful for looping through all channels and not having Event, EKG or Stim channels in averages.
      - accepts list of channels to remove or removes default irrelevant channels
    - **Initialize_epochs_object** – used for creating epochs objects (loads data automatically)
      - accepts channels of interest and any kwargs used by [MNE Epochs](http://martinos.org/mne/stable/generated/mne.Epochs.html)
      - saves epochs object as data.epochs
    - **create_evoked** - used for creating evoked objects
      - accepts "condition" from event_id list (as string)
      - returns evoked object and saves object to data.evoked_list
    - **compute_power** - used to calculate power with [tfr_morlet](http://martinos.org/mne/stable/generated/mne.time_frequency.tfr_morlet.html)
      - accepts "condition" and all kwargs for tfr_morlet
      - out-dated : better to use [tfr_multitaper](http://www.martinos.org/mne/dev/generated/mne.time_frequency.tfr_multitaper.html)
    - **compute_diff_power** - used to calculate ratio of two TF objects
      - accepts two TF objects
      - returns combined (one/two) TF object
  
