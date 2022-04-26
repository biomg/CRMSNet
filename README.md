# CRTNet: a deep learning model that uses convolution and residual transformer block to predict RBPs for RNA sequence  
Computational algorithms for identifying RNAs that bind to specific RBPs are urgently needed, and they can complement high-cost experimental methods.  <br>
In this study, we propose convolutional residual attention mechanism network (CRTNet) that combies CNN, ResNet and residual transformer blocks to predict RNA-protein binding sites for RNA sequences.   <br>

# Dependency:
python 2.7 <br>
Pytorch 1.4.0 <br>

# Data 
Download the trainig and testing data from http://www.bioinf.uni-freiburg.de/Software/GraphProt/GraphProt_CLIP_sequences.tar.bz2 and decompress it in current dir.  <br>
It has 24 experiments of 21 RBPs, and we need train one model per experiment. <br>

# Supported GPUs
Now it supports GPUs. The code support GPUs and CPUs, it automatically check whether you server install GPU or not, it will proritize using the GPUs if there exist GPUs. <br> In addition, CRTNet can also be adapted to protein binding sites on DNAs and identify DNA binding speciticity of proteins.  <br>

# Usage:
python crt.py [-h] [--posi <postive_sequecne_file>] <br>
                 [--nega <negative_sequecne_file>] [--model_type MODEL_TYPE] <br>
                 [--out_file OUT_FILE] [--train TRAIN] [--model_file MODEL_FILE] <br>
                 [--predict PREDICT] [--testfile TESTFILE] [--batch_size BATCH_SIZE] <br>
                 [--motif MOTIF] [--motif_dir MOTIF_DIR]<br>
                 [--num_filters NUM_FILTERS] [--n_epochs N_EPOCHS] <br>
                 
It supports model training, testing. <br>

# Use case:
Take ALKBH5 as an example, if you want to predict the binding sites for RBP ALKBH5. <br>
You first need train the model for RBP ALKBH5, then the trained model is used to predict binding probability of this RBP for your sequences.  <br>
MCNN will train a  model using mutiple CNNs, which are trained using positves and negatives derived from CLIP-seq. <br>

# step 1:
python crt.py 
--posi=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa  <br> 
--nega=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa <br>
--model_type=CRTNet_4 --train=True --n_epochs=40 <br>

CRTNet will save 'model.pkl.101' and 'model.pkl.501' for local and global RNA sequence information, respectively.<br>

# step 2:
python crt.py --testfile=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.positives.fa  <br>
--nega=../GraphProt_CLIP_sequences/ALKBH5_Baltz2012.ls.negatives.fa  <br>
--model_type=CRTNet_4 --predict=True <br>

testfile is your input fasta sequences file, and the predicted outputs for all sequences will be defaulted saved in "prediction.txt". The value in each line corresponds to the probability of being RBP binding site for the sequence in fasta file. The AUC outputs for all sequences will be saved in "pre_auc.txt". The training time and testing time for all sequences will be saved in "time_train.txt" and "time_test.txt". <br>
NOTE:if you have positive and negative sequecnes, please put them in the same sequecne file, which is fed into model for prediciton. <br> 
DO NOT predict probability for positive and negative sequence seperately in two fasta files, then combine the prediction. <br>

# Identify motifs:
You need install WebLogo (http://weblogo.berkeley.edu/) and TOMTOM in MEME Suite(http://meme-suite.org/doc/download.html?man_type=web) to search identifyed motifs against known motifs of RBPs. And also you need has positive and negative sequences when using motif option. <br> 
<br>
# step 3:
python crt.py --posi=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.positives.fa --nega=GraphProt_CLIP_sequences/ALKBH5_Baltz2012.train.negatives.fa --model_type=CRTNet_4 --model_file=model.pkl --motif=True --motif_dir=motifs

The identified motifs (PWMs, and Weblogo) are saved to be defaulted dir motifs (you can also use --motif_dir to configure your dir for motifs), and also include the report from TOMTOM.

# NOTE
When you train CRTNet on your own constructed benchmark dataset, if the training loss cannot converge, may other optimization methods, like SGD or RMSprop can be used to replace Adam in the code.  <br>

# Contact
Zhengsen Pan: zhengsenpan@foxmail.com <br>

# Updates:
26/4/2022 <br>
