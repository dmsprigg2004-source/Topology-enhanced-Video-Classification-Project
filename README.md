# Topology-enhanced Video Classification Project

The purpose of this project is to explore various ways of incorporating topology into a baseline video classification model for enhanced performance.

Current tests implemented:

    - Concatenation based fusion with persistence images
    - Concatenation based fusion with one persistence image for each colour channel
    - A multi-branch architecture separating video frames and their corresponding persistence images which concatenates their predictions at the end of the model
    - A multi-branch architecture that implements Convolutional Block Attention Module (CBAM)

File by file expaination:

baseline_model.py:

    - Tests baseline 3D CNN video classification model

pi_only.py:

    - Inputs persistence images into the baseline 3D CNN to assess topological influence

concatenation_fusion.py:

    - Extracts topological features from data
    - Contains two implementations of concatenation-based feature fusion. One done by separating colour channels and one that keeps them intact

multi_branch_fusion.py:

    - Creates an alternate version of the baseline 3D CNN model that contains two branches. One for video frames, and another for persistence images. These are evaluated by the model separately and then concatenated to create a single output.
    - Contains a secondary test that uses Convolutional Block Attention Module (CBAM)

utils.py:

    Includes several helper functions which together do the following:

        - Load and preprocess video data
        - Implement baseline 3D CNN video classification model
        - Assess model performance with standard classification metrics

CBAM_keras:
    
    - Contains a license crediting original author of CBAM-keras
    - Contains attention_module.py which implements both spatial and channel attention modules (the code was altered to work with 3d tensors)

Confusion_Matrices/History_Plots:

    - Folders to store outputted model assessment plots

CM_Test_Settings:

    - Folder to store Excel files detailing classification metrics and test settings from a test

Test_Results:

    - Folder to store all test result data