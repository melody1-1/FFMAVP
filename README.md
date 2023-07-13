# FFMAVP

Antiviral peptides (AVPs) are widely found in animals and plants, with high specificity and strong sensitivity to drug-resistant viruses. However, due to the great heterogeneity of different viruses, most of the AVPs have specific antiviral activities. Therefore, it is necessary to identify the specific activities of AVPs on virus types. Most existing studies only identify AVPs, with only a few studies identifying subclasses by training multiple binary classifiers. We develop a two-stage prediction tool named FFMAVP that can simultaneously predict AVPs and their subclasses. In the first stage, we identify whether a peptide is AVP or not. In the second stage, we predict the six virus families and eight species specifically targeted by AVPs based on two multiclass tasks. Specifically, the feature extraction module in the two-stage task of FFMAVP adopts the same neural network structure, in which one branch extracts features based on amino acid feature descriptors and the other branch extracts sequence features. Then, the two types of features are fused for the following task. Considering the correlation between the two tasks of the second stage, a multitask learning model is constructed to improve the effectiveness of the two multiclass tasks. In addition, to improve the effectiveness of the second stage, the network parameters trained through the first stage data is used to initialize the network parameters in the second stage. As a demonstration, the cross-validation results, independent test results and the visualization results show that FFMAVP achieves great advantages in both stages.

## How to test sequences？
1. In the first stage of FFMAVP, we first use Feature_Extract_1 to extract features, then save the file and run main_1.py. You need to import the specified file.
2. In the second stage of FFMAVP, we first use Feature_Extract_2 to extract features, then save the file and run main_2.py. You need to import the specified file.
