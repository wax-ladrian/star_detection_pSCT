# star_detection_pSCT

This is the code that can identify and match the starfield in a run/subrun for a pSCT data collection. It uses SEP to extract the point sources, and astroalign to find offset between detected sources and stars from [hsv catalog](https://github.com/astronexus/HYG-Database). It calculates pointing offset as well.

Need to make image cleaning better, and need to agree with some coordinate standard. 

Run the jupyter notebook on Cobalt to access data. Need to use in a run that has R1 files already. The notebook is currently written for 400215. Stick to that one until unblinding.