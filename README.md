# Predicting the presence of the signal peptides in proteins with support vector machine
This repository contains the files generated to complete the final project for the Laboratory of Bioinformatics 2 course of the Master's Degree in Bioinformatics of the Università di Bologna, course 2023-2024, as well as the final report for the project.
This project consists of comparing two different approaches for signal peptide detection, the von Heijne method, a simple method based on motif discovery and recognition, and Support Vector Machines (SVM), a more advanced machine-learning approach.

For details, see the written report (`project-lb2-giacomo_orsini.pdf`).

Written by Giacomo Orsini.

**Table of Contents**
- [Introduction]
  - [Signal peptides]
  - [Methods for signal peptides prediction]
- [Workflow]
  - [0. Requirements and databases](#0-requirements-and-databases)
  - [1. Data collection and analysis](#1-data-collection-and-analysis)
    - [1.1. PDB Search](#11-pdb-search)
    - [1.2. Protein Alignment](#12-protein-alignment)
  - [2. HMM Generation](#2-hmm-generation)
    - [2.1. Train a HMM profile](#21-train-a-hmm-profile)
    - [2.2. Training set](#22-training-set)
  - [3. Method Testing](#3-method-testing)
    - [3.1. Validation set](#31-validation-set)
    - [3.2. Cross-validation and Testing sets](#32-cross-validation-and-testing-sets)
    - [3.3. Model evaluation](#33-model-evaluation)

## Introduction

## Signal peptides
Signal peptides (SP) play a crucial role in cellular processes, serving as guiding tags for proteins destined for secretion or integration into cellular membranes. These short amino acid sequences facilitate the accurate trafficking of proteins within cells, ensuring they reach their intended destinations. Immature proteins directed toward the cell secretory pathway (ER-Golgi-Membrane-Extracellular) are endowed with a signal sequence in the N-terminal region; the signal sequence is cleaved after the protein reaches its final destination. 5 types of signal peptides exist, depending on the signal peptidase that cleaves them and their pathways (SEC or TAT). Recognition of signal peptides is a crucial step for the characterization of protein function and subcellular localization, as well as for deciphering cellular pathways. Moreover, knowledge of protein localization allows to identify potential protein interactors, providing insightful information for drug discovery. Despite their importance, it is challenging to determine whether unknown proteins have signal peptides due to the inherent characteristics of the SPs. These characteristics include the dissimilarities in domain properties, the absence of cleavage site conservation across proteins and species, and the resemblance of the hydrophobic core to other structures such as the transmembrane helix or the transit peptides.
The signal peptide has an average length of 16-30 residues. It has a modular structure comprising three separate regions with different properties: a positively charged region at the N-terminal (1-5 residues long), a hydrophobic core similar to a transmembrane helix (7-15 residues), and a polar C-terminal domain (3-7 residues). The cleavage site is located at the C-terminal and it has an A-X-A motif weakly conserved. The cleavage site allows for the cleavage of the signal peptide to make the protein mature. The biological question of predicting the presence of secretory signal peptides in proteins is a relevant problem of protein function and subcellular localization prediction that has been explored extensively in scientific literature.

## 0. Requirements and used versions and releases
To be able to conduct this project, download the MMseqs2 software from their [GitHub page](https://github.com/soedinglab/MMseqs2). 

It is also necessary to download some Python packages: `pandas`, `Matplotlib`, `seaborn`, `NumPy`, `scikit-learn` and `Biopython`.

All UniProt searches were performed using Release 2023_04.

## 1. Data collection and analysis
The goal of this first section is to retrieve two preliminary processing sets (Positive and Negative) from [UniProt](https://www.uniprot.org/), filter them and split them into training (cross-validation) and benchmarking set.

### 1.1. Retrieve training and benchmarking datasets from UniProtKB
First, access the Advanced search section of the UniProt website, and write the desired query, for . In my case, I opted for these parameters:
- Pfam identifier, SCOP2 lineage identifier, or CATH lineage identifier (`PF00014`, `4003337` or `4.10.410.10`): to select those structurally-resolved PDB structures that have been annotated as containing a Kunitz domain.
- Sequences with no mutations (`Polymer Entity Mutation Count=0`): wild-type versions of the protein, no mutants.
- Resolution (`<= 3`).
- Polymer Entity Sequence Length (`51 - 76` residues): size range of the Kunitz domains.

This is the resulting query for the positive set: 
```
positive: (taxonomy_id:2759) AND (length:[30 TO *]) AND (reviewed:true) AND (ft_signal_exp:*)
```
negative: (reviewed:true) AND (taxonomy_id:2759) NOT (ft_signal:*) AND (length:[30 TO *]) AND ((cc_scl_term_exp:SL-0091) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0173) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0039))

### 1.2. Prepare the datasets for cross-validation
### 1.3. Perform and visualize statistics on both datasets

## Data pre-processing

Prepare your data for training and prediction -> feature extraction
● Implement the von Heijne algorithm
● Implement the SVM-based approach (using sklearn)
● Perform experiments (in cross-validation and blind test) and discuss results
● (Optional) Implement and test different solutions
● Write the manuscript
