# APEX - Molecular de-extinction of antibiotics enabled by deep learning

## Predict AMPs using APEX
By running predict.py, species-specific antmicrobial activties (MICs) of peptides in test_seqs.txt will be generated and saved in Predicted_MICs.csv. To predict antmicrobial activties for novel peptides, you can replace the peptides in test_seqs.txt with the peptides of your interest. Alternatively, you can change line 84 of predict.py to the path of your peptide file. Please make sure that in this file, each line corresponds to a single peptide sequence (<= 50 amino acids in length).


## Software version
pytorch: 1.11.0+cu113


## Contacts
If you have any questions or comments, please feel free to email Fangping Wan (fangping[dot]wan[at]pennmedicine[dot]upenn[dot]edu) and/or César de la Fuente (cfuente[at]pennmedicine[dot]upenn[dot]edu).

