# Design of multimodal antibiotics against intracellular infections using1 deep learning

Pretrained models for manuscript "Design of multimodal antibiotics against intracellular infections using1 deep learning".

## Getting started
(1) Folder MMDVAE contains the pretrained MMD-VAE for peptide sequence generation. To generate peptides around the user-provided prior sequences, replace the example_prior_seqs in line 191 of sampling.py and run it.

(2) Folder APEX contains our antimicrobial peptide predictor (https://doi.org/10.1038/s41551-024-01201-x). 

(3) Folder CPP_predictor contains our pretrained cell-penetrating peptide predictors. To make your own predictions, replace example_list in line 76 of predict_CPPs.py and run it. Results will be saved in CPP_prediction.csv.


