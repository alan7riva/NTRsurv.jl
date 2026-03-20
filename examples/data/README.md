# Datasets used in example tutorials

This folder contains datasets used in the example notebooks for the package.

---

## tongue.csv

Description:
Tongue cancer survival times with censoring indicators with 80 rows. 

Source:
Klein and Moeschberger (1997) Survival Analysis Techniques for Censored and truncated data, Springer. Sickle-Santanello et al. Cytometry 9 (1988): 594-599. Obtained from R package `survival` (LGPL>=2 license).

Modifications:
Omitted `type` column.

---

## lung.csv

Description:
Lung cancer

Modifications:
Changed `time` scale to years by dividing with 365, changed `status` to binary 0-1 variables with 1 denoting an exact observation, dropped covariate columns distinct from `age`, `sex` and `wt_loss`, removed rows with NaN values, changed `age` scale 
by dividing with 100, and standardized `wt_loss` by substracting median followed by diving inter-quantile range.

---

## rossi.csv

Description:
Criminal recidivism data of 432 convicts who were released from Maryland state prisons in the 1970s and who were followed up for one year after release. Half the released convicts were assigned at random to an experimental treatment in which they were given financial aid; half did not receive such aid. 

Source:
Rossi, P.H., R.A. Berk, and K.J. Lenihan (1980). Money, Work, and Crime: Some Experimental Results. New York: Academic Press. Obtained from Julia package `Survival.jl` (MIT license).

...