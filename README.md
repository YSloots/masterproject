# =============================================
# Masterproject Youri Sloots Radboud University
# =============================================
This repository will contain scripts, results and notes for my master internship project at the RU. It contains many scipts that were used to test and fix small bugs,
but also the scripts used to produce the figures and data that are included in the master thesis. This readme serves as a guide to the import scirpts that were
used to produce the results.


# Scripts that are relevant for the results section of the thesis:
- spectral_hardening.py:    Test the capability of distinguishing between two GCRE spectral models in a controled setting
- hiidata_pipeline.py:      Load and use data from the Polderman catalog to constrain models of the GCRE spectrum.
- losdata.py:               Load an plot histograms of data from the Polderman catalog to characterize three key attributes.
- test_data.py:             Test the pipeline inference performance vs three key properties of the Polderman catalog.
- test_geometry.py:         Investigate why the evidence is so sensitive to relative distance error (geometric effect toy example test).
- turbulent_pipeline.py:    Investigate how the inference is affected by a scalable turbulent GMF.
- pipeline_controllers.py:  Gridsearch of three pipeline hyper-parameters for suitable values.
- multiparam_pipeline.py:   Example of inference with many free parameters using our synchrotron-LoS implementation.



