Notes for trying to reproduce power specturm plot_one_slice

Animal experiments:
R6_B10
R6_B16
R32_B7 
R18_B12

Plan 1:
-Start with simulation data.
-Is there preprocessed data?
-If so, run it through power_spectrum.py to calculate spectrums (if needed)
-Plot and compare with 4F
-Then move to animal data.
-I think that the all preprocessed data lives on Vyassa's machine. I tried to copy open but ran into an error
    - Try again
    - If unable to get preprocessed data due to permission issue, recreate preprocessed data via process_nwb
- Then run through power_spectrumpy
- Compare with 4F
- If all is good, then calculate pairware correlation between animal and simulation power spectrums
- Plot histogram

Requires pynwb==1.3.1 -- add in requirement doc 
Add code to save spectrum data to disk

Save correlation data to nwb file in: 
/processing/Hilb_54bands/ECoG/spectrum: 2D array power for frequency x channels
/processing/Hilb_54bands/ECoG/frequency: 1D array frequency values

Have written grand spectrum average for one channel to nwb
But spectra for each trial presentation is plotted. Does it matter if we do correlation between experimental channels and simulation average or experimental channels and individual spectra trials?

Spectra for experiments found!

Now compute correlation between sim and experimental!