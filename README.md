# Predicting the presence of the signal peptides in proteins with support vector machine
This repository contains the files created for the final project for the Laboratory of Bioinformatics 2 course of the Master's Degree in Bioinformatics of the Università di Bologna, course 2023-2024, as well as the final report for the project.
This project consisted of implementing and comparing two different approaches for signal peptide detection: the von Heijne method, a simple method based on motif discovery and recognition, and Support Vector Machines (SVM), a more advanced machine-learning approach.

For details, see the written report (`project-lb2-giacomo_orsini.pdf`).

Written by Giacomo Orsini.

# Abstract
Signal Peptides (SP) play a crucial role in the secretory pathway of cells. In-silico prediction of signal peptides offers a valuable solution to the biological problem of identifying proteins implied in this pathway, a significant task for scientific efforts such as drug discovery. Many computational methods for prediction have been proposed in the literature, with different features and performances. In the hereby presented experiment, two predictors for signal peptides were constructed using the Von Heijne algorithm and the machine learning method of Support Vector Machine. The models were trained and tested using datasets of reviewed proteins, containing examples of both signal peptide-endowed proteins and proteins without it. A Cross-Validation step has been added in both workflows. The performances of the two final models were compared, showing that the SVM model is a better classifier overall than the Von Heijne model.

# Index
**Table of Contents**
- [Introduction](#introduction)
  - [Signal peptides](#signal-peptides)
  - [Methods for signal peptides prediction](#signal-peptide-prediction)
- [Workflow](#workflow)
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

# Introduction

## Signal peptides
**Signal peptides** (SP) play a crucial role in cellular processes, serving as guiding tags for proteins destined for secretion or integration into cellular membranes. These short amino acid sequences facilitate the accurate trafficking of proteins within cells, ensuring they reach their intended destinations. Immature proteins directed toward the cell secretory pathway (ER-Golgi-Membrane-Extracellular) are endowed with a signal sequence in the N-terminal region; the signal sequence is cleaved after the protein reaches its final destination. 5 types of signal peptides exist, depending on the signal peptidase that cleaves them and their pathways (SEC or TAT). Recognition of signal peptides is a crucial step for the characterization of protein function and subcellular localization, as well as for deciphering cellular pathways. Moreover, knowledge of protein localization allows the identification of potential protein interactors, providing insightful information for drug discovery. Despite their importance, it is challenging to determine whether unknown proteins have signal peptides due to the inherent characteristics of the SPs. These characteristics include the dissimilarities in domain properties, the absence of cleavage site conservation across proteins and species, and the resemblance of the hydrophobic core to other structures such as the transmembrane helix or the transit peptides.

The signal peptide has an **average length of 16-30 residues**. It has a modular structure comprising three separate regions with different properties: 
- a **positively charged** region at the N-terminal (1-5 residues long)
- a **hydrophobic core** similar to a transmembrane helix (7-15 residues)
- a **polar C-terminal domain** (3-7 residues). The cleavage site is located at the C-terminal, and it has an A-X-A motif that is weakly conserved.

The **cleavage site** allows for the cleavage of the signal peptide to make the protein mature. The biological question of predicting the presence of secretory signal peptides in proteins is a relevant problem of protein function and subcellular localization prediction that has been explored extensively in scientific literature.

![image](https://github.com/user-attachments/assets/0c1980c8-2325-49c4-bf93-76bf78037712)

## Signal peptide prediction
Despite their biological importance, experimentally identifying signal peptides can be challenging and resource-intensive. This has led to the development of computational methods, which offer a more efficient and cost-effective approach, opening a new set of challenges. Available in-silico methods can be divided into three main classes:

- **Homology-based approaches**. In these approaches, information is transferred from closely related sequences. They can be accurate but require a good template, experimentally studied.
- **Algorithmic and probabilistic approaches**. A traditional algorithm takes some input and some logic in code and drums up the output. In this category, methods such as Hidden Markov Models and Weight matrix approaches can be found.
- **Machine learning approaches**. A Machine Learning Algorithm takes an input and an output and gives the logic that connects them, which can then be used to work with new input to give one an output. Typical Machine learning methods are Artificial Neural Networks (including recent deep-learning methods), Decision trees and Random Forests, and Support Vector Machines.

The problem of signal peptide prediction (and more in general motives prediction) has two dimensions: the correct identification of protein containing the motif from proteins not containing it and the labelling of the motif inside the protein sequence.

### Von Heijne Algorithm
The first method specific for signal peptide prediction was introduced by Gunnar Von Heijne in 1983 (Von Heijne, 1983). The **Von Heijne method** was based on a reduced-alphabet weight matrix combined with a rule for narrowing the search region; only seven different weights at each position, corresponding to groups of amino acids (AAs) with similar properties, were estimated manually (rather than using automatic procedures). The matrix score aimed to recognize the location of the signal peptide (SP) cleavage site. The scores were computed for positions 12-20, counted from the beginning of the hydrophobic region (defined as the first quadruplet of AAs with at least three hydrophobic residues). In the first benchmark, the procedure correctly predicted 92% of sites on data used to estimate it, while on a later benchmark, the test performance was only 64% (a sign of overfitting).

Since its first publication, the Von Heijne method was updated by introducing a more structured weight matrix (that uses log odds scores), pseudo counts to mitigate the sampling error (the fact that the model is constructed from a limited set of examples), and a longer window of residues for the search of the cleavage site (40-50 residues) (Von Heijne, 1986). This updated version of the method was used in the experiment (with a window of 90 residues).

### Neural Networks
After the increase of interest in this machine learning method in the 80s, **Neural Networks** (NN) have affirmed themselves as the most successful framework for predicting signal peptides (SPs) and their cleavage site. Early approaches used a shallow networks approach (few hidden layers): a fixed-size portion of the N-terminus (e.g., the first 20 residues) was given as input to the NN; a sliding window approach was implemented to scan both for the presence of the signal peptide and cleavage site position (identification and labelling). Tools that used this approach are **SignalP1-4** (Nielsen et al., 1997; Nielsen and Krogh, 1998; Bendtsen et al., 2004; Petersen et al., 2011) and **SPEPLip** (Fariselli et al., 2003).

Novel approaches have progressively evolved toward the use of more complex and deeper network architectures. Examples of tools are **DeepSig** (Savojardo et al., 2018) and **SignalP5-6** (Almagro Armenteros et al., 2019; Teufel et al., 2022). These deep learning methods have nowadays affirmed themselves as gold standards; for example, SignalP5-6 has been demonstrated to be capable of predicting all 5 types of signal peptides.

### Support Vector Machines
**Support Vector Machines** (SVM) are powerful machine-learning algorithms that can be used for classification and regression. Briefly, SVMs work by finding a hyperplane in the feature space that separates the data points into two classes with the largest possible margin. This hyperplane is then used to classify new data points. Although less popular than NN, SVMs have been used to predict signal peptides with great success. SVMs have been shown to be very accurate at predicting signal peptides (Vert, 2001).

# Workflow
A proper dataset of proteins comprising signal peptide proteins and non-signal peptides (positives and negatives) has been constructed starting from data fetched on the Swiss-Prot database; to avoid biases, a data preprocessing step with numerous quality checks has been performed. The dataset has been divided into two sets, the **Training set** and the **Benchmarking set**; the Training set has been further divided into **5 Cross Validation sets** (CVs). Following the creation of the datasets, the first method implemented has been the Von Heijne algorithm. Briefly, the steps to implement this method have been:

1. Training of 5 different models using different cross-validation subsets combinations.
2. Extraction of an optimal threshold to discriminate positives and negatives by testing the models on a validation set.
3. Performance evaluation of the models on a testing set.
4. Average optimal threshold computation and training of a final model using the entire Training set.
5. Performance evaluation of the final model tested on the Benchmarking set, with the average threshold as discriminatory.

The second method implemented has been the machine learning algorithm of Support Vector Machines in the form of a binary classifier (SVC). Briefly, the steps to implement this method have been:

1. Discriminatory feature extraction and input encoding of data.
2. Training of different models using different combinations of hyperparameters, features, and Cross-validation sets.
3. For each feature combination, the best model selection is by testing the models on validation sets, saving the most common hyperparameters and the best scoring set of Training sets.
4. Performance evaluation of the best model on the testing set.
5. Selection of the best set of features and creation of a final model, trained on the Training set and tested on the Benchmarking set.
6. The final results, as well as the partial results of the two methods, have been compared. A standard set of metrics has been used for the performance evaluation. Finally, eventual misclassifications (False Negatives and False Positives) have been examined.

The experiment has been designed as explained to answer two main questions: the biological question of signal peptide prediction, with the creation of a well-performing predictive model and the comparison of different models to find the best one.

# 0. Requirements and used versions and releases
To be able to conduct this project, download the MMseqs2 software from their [GitHub page](https://github.com/soedinglab/MMseqs2). 

It is also necessary to download some Python packages: `pandas`, `Matplotlib`, `seaborn`, `NumPy`, `scikit-learn` and `Biopython`.

All UniProt searches were performed using **Release 2023_04**.

## 1. Data collection and processing
The goal of this first section is to retrieve two preliminary processing sets (Positive and Negative) from [UniProt](https://www.uniprot.org/)(Release 2023_04), filter them and split them into training (cross-validation) and benchmarking set. The creation of representative and unbiased datasets is the first step for every machine learning method and involves both data collection and data preprocessing.

We’ll focus only on Eukaryotes, neglecting bacterial and archaea proteins.

### 1.1. Retrieve training and benchmarking datasets from UniProtKB
First, access the `Advanced search` section of the UniProt website and write the desired query to retrieve the proteins. In my case, I opted for these parameters:

- Pfam identifier, SCOP2 lineage identifier, or CATH lineage identifier (`PF00014`, `4003337` or `4.10.410.10`): to select those structurally-resolved PDB structures that have been annotated as containing a SP.
- Sequences with no mutations (`Polymer Entity Mutation Count=0`): wild-type versions of the protein, no mutants.
- Resolution (`<= 3`).
- Polymer Entity Sequence Length (`51 - 76` residues): size range of the Kunitz domains.

This is the resulting query for the positive set: 
```
positive: (taxonomy_id:2759) AND (length:[30 TO *]) AND (reviewed:true) AND (ft_signal_exp:*)
```
This is the resulting query for the negative set: 
```
negative: (reviewed:true) AND (taxonomy_id:2759) NOT (ft_signal:*) AND (length:[30 TO *]) AND ((cc_scl_term_exp:SL-0091) OR (cc_scl_term_exp:SL-0191) OR (cc_scl_term_exp:SL-0173) OR (cc_scl_term_exp:SL-0209) OR (cc_scl_term_exp:SL-0204) OR (cc_scl_term_exp:SL-0039))
```
In my case, the **preliminary positive set** comprises 2969 entries, while the **preliminary negative set** 31619.
Once the data are retrieved, we can download a perto in `.tsv` format and store them in a folder.
```
mkdir SP_model | cd ./SP_model | mkdir datasets | cd datasets

gunzip uniprot_prel_pos_set.tsv.gz 
gunzip uniprot_prel_neg_set.tsv.gz
```
### 1.2 Quality checks
Positive and Negative raw data retrieved in the previous step suffer from biases that must be corrected with a proper preprocessing procedure. Specifically, we have to filter out missclassified proteins, inadequate sequences and redundant proteins.

#### 1.2.1 Filtering
From the preliminary Positive set, the entries with a non-defined signal peptide (marked with a "?" symbol) and the ones with a peptide smaller than 13 residues have to be removed, as the length of the cleavage site of a signal peptide is 13 residues. From the preliminary Negative set, entries with a subcellular location in the Endoplasmic Reticulum, Golgi Apparatus, Lysosome or secreted have to be removed, as signal peptides can be found in these organelles and have this property.

```
#Positive set
grep -v '?' uniprot_prel_pos_set.tsv | awk -F "\t" '{split($2, e, /[.;]/); if (e[3]>=13) print $1"\t"e[3]}' >uniprot_pos_set.tsv

#Negative set
grep -v -i 'endoplasmic\|golgi\|lysosome\|secreted' uniprot_prel_neg_set.tsv > uniprot_neg_set.tsv

#remove the entry line from the files
grep -v 'Entry' uniprot_neg_set.tsv | cut -f 1 >uniprot_neg_set_ID.txt
grep -v 'Entry' uniprot_pos_set.tsv | cut -f 1 >uniprot_pos_set_ID.txt
```
After this step, the **cleaned positive set** contained 2942 entries and the **cleaned negative set** 30011.

#### 1.2.2 Retrieve FASTA sequences
Now, extract the Uniprot IDs from these files and use them to retrieve the FASTA files with the `UniProt ID mapping tool`. Download and save the data in the same folder. This step is necessary to remove redundant proteins with MMseq2.

#### 1.2.3 Remove Redundancy
The query search procedure from SwissProt ensures the fetching of all proteins with the desired features; hence, the chance of having proteins with similar or identical sequences is very high. This leads to a potential redundancy bias that can harm the creation of predictors. To ensure the absence of redundancy in the Positive and Negative sets, the **MMSeqs2** can be used to cluster proteins of each set and extract representatives. 
MMSeq2 (Many-against-Many sequence searching) is a software suite to search and cluster protein and nucleotide sequence sets. A single run produces many files, among which two are important to us: `cluster results_rep_seq.fasta`, a FASTA file containing all the representative sequences, one for each found cluster, and `cluster-results_cluster.tsv`, reporting the IDs of each sequence and their cluster’s representatives. Here the commands I used:
```
#clusterization
mmseqs easy-cluster uniprot_pos_set.fasta mmseq_pos_set tmp --min-seq-id 0.3 -c 0.4 --cov-mode 0 --cluster-mode 1
mmseqs easy-cluster uniprot_neg_set.fasta mmseq_neg_set tmp --min-seq-id 0.3 -c 0.4 --cov-mode 0 --cluster-mode 1
```
Where:
- `--min-seq-id`:`0.3` means that the sequence identity threshold is 30%.
- `-c`:`0.4` means coverage threshold is 40%.
- `--cov-mode`:`0` refers to the coverage computation mode.
- `--cluster-mode`:`1` refers to the cluster structure, like connected component, single linkage etc.

Now extract the IDs and randomize them to then create the training and benchmarking sets:
```
grep '^>' mmseq_pos_set_rep_seq.fasta | cut -d '|' -f 2 | sort -R > mmseq_pos_set_ID_rdm.txt
grep '^>' mmseq_neg_set_rep_seq.fasta | cut -d '|' -f 2 | sort -R > mmseq_neg_set_ID_rdm.txt
```
At the end of this step, my positive set contains 1093 IDs and the negative 9523 IDs.

## 2. Training and Benchmarking sets
The data processing allows us to construct an unbiased and well balanced **Training set** and a **Benchmarking set**.
- Training set: it is used to train the methods, optimize model hyperparameters and perform Cross-Validation experiments.
- Benchmarking set: the **holdout data set** is used to test the generalization performance of the different models.

### 2.1 80/20 split
The proportion of the examples contained in the Training and Benchmarking set should be 80:20. In practice, 80% of the positive set IDs and negative set IDs have to go in the Training set and 20% in the Benchmarking set.

```
head -n 874 mmseq_pos_set_ID_rdm.txt > training_set_pos.txt
head -n 7618 mmseq_neg_set_ID_rdm.txt > training_set_neg.txt
tail -n +875 mmseq_pos_set_ID_rdm.txt > benchmarking_set_pos.txt
tail -n +7619 mmseq_neg_set_ID_rdm.txt > benchmarking_set_neg.txt

cat benchmarking_set_pos.txt benchmarking_set_neg.txt > benchmarking_set.txt
cat training_set_pos.txt training_set_neg.txt > training_set.txt
```

The Training and Benchmarking sets contain, respectively, 8492 and 2124 entries.

### 2.2 Retrieve metadata
The newly created datasets in contain just the UniProt IDs. The UniProt ID Mapping Tool is necessary now to retrieve numerous useful metadata, that we will use to do statistical analysis, downloadable in a `.tsv` format:
- Signal peptide length
- Protein length
- Taxonomic lineage (to extract Kingdom)
- Organism (to extract species)
- Protein sequence

Extra "shell" steps allow us to add an extra column containing the class of each protein in binary form (0 negatives and 1 positives). The final `.tsv` file has the following columns:
- Entry name
- Signal peptide length (NaN if not present)
- Protein length
- Kingdom
- Species
- Protein sequence
- Class

```
#Extract from the tsv file only the desired properties (entry name, SP length, protein length, kingdom, species, sequence)
cat <(echo -e "Accession code\tSP length\tLength\tKingdom\tSpecies\tSequence") <(awk -F '\t' '{split($3, e, /[.;]/); split($5, k, /[, ]/); split($6, s, /[ ]/); if (k[10]=="(kingdom)") print $1"\t"e[3]"\t"$4"\t"k[9]"\t"s[1], s[2]"\t"$7; else if (k[14]=="(kingdom)") print $1"\t"e[3]"\t"$4"\t"k[13]"\t"s[1], s[2]"\t"$7; else print $1"\t"e[3]"\t"$4"\tOther\t"s[1], s[2]"\t"$7'} training_set.tsv | tail -n +2 > training_set_parsed.tsv
cat <(echo -e "Accession code\tSP length\tLength\tKingdom\tSpecies\tSequence") <(awk -F '\t' '{split($3, e, /[.;]/); split($5, k, /[, ]/); split($6, s, /[ ]/); if (k[10]=="(kingdom)") print $1"\t"e[3]"\t"$4"\t"k[9]"\t"s[1], s[2]"\t"$7; else if (k[14]=="(kingdom)") print $1"\t"e[3]"\t"$4"\t"k[13]"\t"s[1], s[2]"\t"$7; else print $1"\t"e[3]"\t"$4"\tOther\t"s[1], s[2]"\t"$7'} benchmarking_set.tsv | tail -n +2 > benchmarking_set_parsed.tsv

#add a column to the tr and bm tsv files containing the class of each protein (1 have SP, 0 don"t have sp)
#training set
awk -F '\t' '{if ($2 == "") print $2"\t"$6"\t0""\t"$1}' training_set_parsed.tsv > training_set_parsed_negc.tsv
awk -F '\t' '{if ($2 != "") print $2"\t"$6"\t1""\t"$1}' training_set_parsed.tsv > training_set_parsed_posc.tsv
#benchmarking set
awk -F '\t' '{if ($2 == "") print $2"\t"$6"\t0""\t"$1}' benchmarking_set_parsed.tsv > benchmarking_set_parsed_negc.tsv
awk -F '\t' '{if ($2 != "") print $2"\t"$6"\t1""\t"$1}' benchmarking_set_parsed.tsv > benchmarking_set_parsed_posc.tsv
#concatenate the benchmarking subsets
cat benchmarking_set_parsed_negc.tsv benchmarking_set_parsed_posc.tsv > benchmarking_set_parsed_totc.tsv
cat training_set_parsed_negc.tsv training_set_parsed_posc.tsv > training_set_parsed_totc.tsv
```

### 2.3. Cross-validation sets
A **Cross-Validation** procedure is crucial to evaluating the model design and avoiding potential biases such as overfitting; validating the model on a limited amount of data allows for a less optimistic estimate of how well the model will generalize, also helps in the choice of the hyperparameters.

To maintain the correct proportion of negative and positive entries in each cross-validation set, the Training set is divided into a **parsed Positive Training set** (874 entries) and a **parsed Negative Training set** (7618 entries). Each of the two sets is then divided into 5 subsets, leading to **5 Cross Validation positive subsets** and **5 Cross Validation negative subsets**. The positive and negative cross-validation subsets are then paired and concatenated into **5 cross-validation sets**, containing CV0, CV1, CV2, and CV3 with 1699 entries and CV4 with 1697 entries, respectively.

```
split -d -a 1 -l$(( ($(wc -l < training_set_parsed_negc.tsv) +4)/ 5)) training_set_parsed_negc.tsv CV_neg_
split -d -a 1 -l$(( ($(wc -l < training_set_parsed_posc.tsv) +4)/ 5)) training_set_parsed_posc.tsv CV_pos_
#make the directory
cd .. | mkdir VH | mkdir VH/CVs | cp ./datasets/CV_neg_* ./VH/CVs | cp cp ./datasets/CV_neg_* ./VH/CVs | cd ./VH/CVs 
#concatenate the negative and positive CV subsets
for i in {0..4}; do cat CV_pos_$i CV_neg_$i > CV$(($i+1)).txt; done     #CV$(($i+1)).txt
cd .. | mkdir ./SETs | cp ../datasets/training_set_parsed_posc.tsv ./SETs |  cp ../datasets/benchmarking_set_parsed_totc.tsv ./SETs 
```

## 3. (OPTIONAL) Statistical analysis of the datasets
To visualize your data and help yourself contextualize the problem, as well as detect unforseen bias, I suggest you do a statistical analysis of the data. As we dowloaded the metadata from UniProt, we are able to conduct the following analysis. The details and explanations of each analysis can be appreciated in the final report and the supplementary material () while the commands used for the analysis in the Jupyter Notebook (). Here I will just show some of the results.

- Signal peptide length distribution. The length has been normalized to a log10 scale.
- Protein length distribution. The length has been normalized to a log10
 scale.
- Taxonomical classification.
- Amino acidic composition of the signal peptides compared to a background SwissProt distribution (amino acidic frequencies considering all the entries in the database), also considering the amino acid properties (UniProtKB/Swiss-Prot, 2023).
- Sequence logos of the cleavage site (with the use of the online tool WebLogo) of the signal peptide.

![image](https://github.com/user-attachments/assets/1eee1e78-8652-44c5-a315-e0d903e3de62)
![image](https://github.com/user-attachments/assets/01959d65-d1c4-42ed-9ae4-9e57f8dab2ed)

### 1.3. Perform and visualize statistics on both datasets

## 4. Von Heijne method
