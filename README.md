# Predicting the presence of the signal peptides in proteins with support vector machine
This repository contains the files generated to complete the final project for the Laboratory of Bioinformatics 2 course of the Master's Degree in Bioinformatics of the Università di Bologna, course 2023-2024, as well as the final report for the project.
This project consists of comparing two different approaches for signal peptide detection, the von Heijne method, a simple method based on motif discovery and recognition, and Support Vector Machines (SVM), a more advanced machine-learning approach.

For details, see the written report (`project-lb2-giacomo_orsini.pdf`).

Written by Giacomo Orsini.

# Abstract
Signal Peptides (SP) play a crucial role in the secretory pathway of cells. In-silico prediction of signal peptides offers a valuable solution to the biological problem of identifying proteins implied in this pathway, a significant task for scientific efforts such as drug discovery. Many computational methods for prediction have been proposed in literature, with different features and performances. In the hereby presented experiment, two predictors for signal peptides were constructed using the Von Heijne algorithm and the machine learning method of Support Vector Machine. A Cross Validation step has been added in both workflows. The models were trained and tested using datasets of reviewed proteins, containing examples of both signal peptide-endowed proteins and proteins without it. The performances of the two final models were compared. Performance evaluation has shown that the SVM model is a better classifier overall than the Von Heijne model.

# Index
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
If you already know what a signal peptide is, skip to the workflow section.

### Signal peptides
Signal peptides (SP) play a crucial role in cellular processes, serving as guiding tags for proteins destined for secretion or integration into cellular membranes. These short amino acid sequences facilitate the accurate trafficking of proteins within cells, ensuring they reach their intended destinations. Immature proteins directed toward the cell secretory pathway (ER-Golgi-Membrane-Extracellular) are endowed with a signal sequence in the N-terminal region; the signal sequence is cleaved after the protein reaches its final destination. 5 types of signal peptides exist, depending on the signal peptidase that cleaves them and their pathways (SEC or TAT). Recognition of signal peptides is a crucial step for the characterization of protein function and subcellular localization, as well as for deciphering cellular pathways. Moreover, knowledge of protein localization allows to identify potential protein interactors, providing insightful information for drug discovery. Despite their importance, it is challenging to determine whether unknown proteins have signal peptides due to the inherent characteristics of the SPs. These characteristics include the dissimilarities in domain properties, the absence of cleavage site conservation across proteins and species, and the resemblance of the hydrophobic core to other structures such as the transmembrane helix or the transit peptides.

The signal peptide has an average length of 16-30 residues. It has a modular structure comprising three separate regions with different properties: 
- a positively charged region at the N-terminal (1-5 residues long)
- a hydrophobic core similar to a transmembrane helix (7-15 residues)
- a polar C-terminal domain (3-7 residues). The cleavage site is located at the C-terminal and it has an A-X-A motif weakly conserved.

The cleavage site allows for the cleavage of the signal peptide to make the protein mature. The biological question of predicting the presence of secretory signal peptides in proteins is a relevant problem of protein function and subcellular localization prediction that has been explored extensively in scientific literature.

### Signal peptide prediction
Despite their biological importance, experimentally identifying signal peptides can be challenging and resource-intensive. This has led to the development of computational methods, which offer a more efficient and cost-effective approach, opening a new set of challenges. Available in-silico methods can be divided into three main classes:

- Homology-based approaches. In these approaches, information is transferred from closely related sequences. They can be accurate but require a good template, experimentally studied.
- Algorithmic and probabilistic approaches. A traditional algorithm takes some input and some logic in code and drums up the output. In this category, methods such as Hidden Markov Models and Weight matrix approaches can be found.
- Machine learning approaches. A Machine Learning Algorithm takes an input and an output and gives the logic that connects them, which can then be used to work with new input to give one an output. Typical Machine learning methods are Artificial Neural Networks (including recent deep-learning methods), Decision trees and Random Forests, and Support Vector Machines.

The problem of signal peptide prediction (and more in general motives prediction) has two dimensions: the correct identification of protein containing the motif from proteins not containing it and the labelling of the motif inside the protein sequence.

#### Von Heijne Algorithm
The first method specific for signal peptide prediction was introduced by Gunnar Von Heijne in 1983 (Von Heijne, 1983). The Von Heijne method was based on a reduced-alphabet weight matrix combined with a rule for narrowing the search region; only seven different weights at each position, corresponding to groups of amino acids (AAs) with similar properties, were estimated manually (rather than using automatic procedures). The matrix score aimed to recognize the location of the signal peptide (SP) cleavage site. The scores were computed for positions 12-20, counted from the beginning of the hydrophobic region (defined as the first quadruplet of AAs with at least three hydrophobic residues). In the first benchmark, the procedure correctly predicted 92% of sites on data used to estimate it, while on a later benchmark, the test performance was only 64% (a sign of overfitting).

Since its first publication, the Von Heijne method was updated by introducing a more structured weight matrix (that uses log odds scores), pseudo counts to mitigate the sampling error (the fact that the model is constructed from a limited set of examples), and a longer window of residues for the search of the cleavage site (40-50 residues) (Von Heijne, 1986). This updated version of the method was used in the experiment (with a window of 90 residues).

#### Neural Networks
After the increase of interest in this machine learning method in the 80s, Neural Networks (NN) have affirmed themselves as the most successful framework for predicting signal peptides (SPs) and their cleavage site. Early approaches used a shallow networks approach (few hidden layers): a fixed-size portion of the N-terminus (e.g., the first 20 residues) was given as input to the NN; a sliding window approach was implemented to scan both for the presence of the signal peptide and cleavage site position (identification and labelling). Tools that used this approach are SignalP1-4 (Nielsen et al., 1997; Nielsen and Krogh, 1998; Bendtsen et al., 2004; Petersen et al., 2011) and SPEPLip (Fariselli et al., 2003).

Novel approaches have progressively evolved toward the use of more complex and deeper network architectures. Examples of tools are DeepSig (Savojardo et al., 2018) and SignalP5-6 (Almagro Armenteros et al., 2019; Teufel et al., 2022). These deep learning methods have nowadays affirmed themselves as gold standards; for example, SignalP5-6 has been demonstrated to be capable of predicting all 5 types of signal peptides.

#### Support Vector Machines
Support Vector Machines (SVM) are powerful machine-learning algorithms that can be used for classification and regression. Briefly, SVMs work by finding a hyperplane in the feature space that separates the data points into two classes with the largest possible margin. This hyperplane is then used to classify new data points. Although less popular than NN, SVMs have been used to predict signal peptides with great success. SVMs have been shown to be very accurate at predicting signal peptides (Vert, 2001).

## Workflow

A proper dataset of proteins comprising signal peptide proteins and non-signal peptides (positives and negatives) has been constructed starting from data fetched on the Swiss-Prot database; to avoid biases, a data preprocessing step with numerous quality checks has been performed. The dataset has been divided into two sets, the Training set and the Benchmarking set; the Training set has been further divided into 5 Cross Validation sets (CVs).

The first method used has been the Von Heijne algorithm, an algorithmic method based on position-specific matrices. Briefly, the steps to implement this method have been:

- Training of 5 different models using different Cross Validation subsets combinations.
- Extraction of an optimal threshold to discriminate positives and negatives by testing the models on a validation set.
- Performance evaluation of the models on a testing set.
- Average optimal threshold computation and training of a final model using the entire Training set.
- Performance evaluation of the final model tested on the Benchmarking set, with the average threshold as discriminatory.

The second method used has been the machine learning algorithm of Support Vector Machines in the form of a binary classifier (SVC). Briefly, the steps to implement this method have been:

- Discriminatory feature extraction and input encoding of data.
- Training of different models using different combinations of hyperparameters, features, and Cross validation sets.
- For each feature combination, the best model selection is by testing the models on validation sets, saving the most common hyperparameters and the best scoring set of Training sets.
- Performance evaluation of the best model on the testing set.
- Selection of the best set of features and creation of a final model, trained on the Training set and tested on the Benchmarking set.
- The final results, as well as the partial results of the two methods, have been compared. A standard set of metrics has been used for the performance evaluation. Finally, eventual misclassifications (False Negatives and False Positives) have been examined.

The experiment has been designed as explained to answer two main questions: the biological question of signal peptide prediction, with the creation of a well performing predictive model, and the comparison of different models to find the best one.

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
