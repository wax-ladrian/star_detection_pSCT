# star_detection_pSCT

This is the code that can identify and match the starfield in a run/subrun for a pSCT data collection. It uses [SEP](https://sep.readthedocs.io/en/stable/) to extract the point sources, and [astroalign](https://astroalign.quatrope.org/en/latest/) to find offset between detected sources and stars from [hsv catalog](https://github.com/astronexus/HYG-Database). It calculates pointing offset as well, currently in pixel dimensions

Need to make image cleaning better, and need to agree with some coordinate standard. 

Run the jupyter notebook on Cobalt to access data. Need to use in a run that has R1 files already. The notebook is currently written for 400215. Stick to that one until unblinding.

One can run 'python psct_starfinder_pointing.py -r 400215 -si' where -r is the run, and -si is to save image plots. 

In the settings.yaml, the following options need to be placed:
analysis_options:
  animation_path : path to where animation goes, ending in /, like 'path/to/directory/'
  npys_path : path to where the .npy files are placed, ending in /, like 'path/to/directory/'
  point_data_path : path to where the .csv with the final found positions is saved, ending in /, like 'path/to/directory/'
  P_step : number of seconds a star frame encompasses. 60 s is fine for now
  dpi_imgs : dpi of the plots
  log_files : "/data/wipac/CTA/targetcdata/run{0}_log.log"
  subrun_files : "/data/wipac/CTA/targetcdata/run{0}_subrun{1}_r1.tio"
  catalog_path : repo/path/hyg_v42.csv"
  telescope_config : "repo/path/psct_config.txt"
  run_info_path : 'repo/path/pSCT_Run_Log-DR8_2026.csv'

bad_srs: # you must list which subruns that must be ignored per run. If the r1 file doesn't exist, then it will already ignore it, since it globs.
  400213 : [0]
  400215 : [52, 54, 55]
