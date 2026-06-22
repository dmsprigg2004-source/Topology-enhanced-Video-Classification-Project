# Topology-enhanced Video Classification Project

File by file expaination:

baseline_model.py:

    - Tests baseline 3D CNN video classification model

concatenation_fusion.py:

    - Extracts topological features from data
    - Contains two implimentations of concatenation-based feature fusion. One done by separating colour channels and one that keeps them intact

utils.py:

    Includes several helper functions which together can do the following:

        - Load and preprocess video data
        - Impliment baseline 3D CNN video classification model
        - Assess model performance with standard classification metrics

Confusion_Matrices/History_Plots:

    - Folders to store outputted model assessment plots

