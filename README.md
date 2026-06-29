# Topology-enhanced Video Classification Project

The purpose of this project is to explore various ways of incorporating topology into a baseline video classification model for enhanced performance.

Current tests implemented:

    - Concatenation based fusion with persistence images
    - Concatenation based fusion with one persistence image for each colour channel
    - A multi-branch architecture separating raw video frames and persistence images which averages their predictions at the end of the model
    - A multi-branch architecture that implements Convolutional Block Attention Module (CBAM)

File by file expaination:

baseline_model.py:

    - Tests baseline 3D CNN video classification model

concatenation_fusion.py:

    - Extracts topological features from data
    - Contains two implimentations of concatenation-based feature fusion. One done by separating colour channels and one that keeps them intact

multi_branch_fusion.py:

    - Creates an alternate version of the baseline 3D CNN model that contains two branches. One for raw image frames, and another for persistence images. These are evaluated by the model separately and then averaged to create a single output.
    - Contains a secondary test that uses Convolutional Block Attention Module (CBAM)

utils.py:

    Includes several helper functions which together can do the following:

        - Load and preprocess video data
        - Impliment baseline 3D CNN video classification model
        - Assess model performance with standard classification metrics

CBAM_keras:
    
    - Contains a license crediting original author of CBAM-keras
    - Contains attention_module.py which implements both spatial and channel attention modules (the code was altered to work with 3d tensors)

Confusion_Matrices/History_Plots:

    - Folders to store outputted model assessment plots

