# FLSimulation
## Introduction
This code repository contains the PyTorch Federated Learning Simulation code for the paper 'Federated Learning in Healthcare using Classical Machine Learning Algorithms'. To highlight the importance of traditional ML and shallow learning on small sample sizes in a federated setting, we compare the predictive performance of a Federated Random Forest, a shallow implementation of a Federated Deep Regression Forest, and a deep neural network (DNN) on a single cell line of NCI-60 for drug response prediction with a varying number of samples available to the federation. We also investigate the impact of client heterogeneity on the different federated models by simulating client target distribution shifts. This package has been adapted from [FederatedDeepRegressionForests](https://github.com/DanielNolte/FederatedDeepRegressionForests).

In this package, we provide a PyTorch implementation for training and testing Random Forests, Deep Regression Forests, and deep neural networks in the federated and centralized setting, as well as parameters for adjusting the number of samples, the client skewness, and various training parameters.
   
## Dependencies
The code has been tested in Windows 10 with Python 3.8 or greater and the following packages:
* PyTorch >= 1.12.1
* NumPy >= 1.23.5
* pandas >= 1.5.0
* scikit-learn >= 1.2.2
* CUDA (CPU implementation not tested)

## NCI-60
NCI-60 is an anticancer drug screening dataset maintained by the US National Cancer Institute (NCI) since the late 1980s. Now the dataset contains over 100 cancerous cell lines screened on over 55,000 unique compounds. The response of a compound on a particular cell line is recorded as GI-50, the drug concentration resulting in 50% inhabitation of cell proliferation. We followed the same data extraction as described in the supplementary material of (Nolte et al., [2023](https://doi.org/10.1093/bioadv/vbad036)). To summarize, we used the PaDEL software package to extract a set of 672 chemical descriptors for each drug which were used as the predictors or features. The negative log of GI-50 was used as the target for prediction since the dose administration protocol is logarithmic in nature. (Shoemaker, [2006](https://doi.org/10.1038/nrc1951)). For illustration purposes, we selected the cell line with the most screened compounds with 50857 samples/compounds (A549) and considered a varying number of samples available to the federation. 
## Models
The federated random forest model was implemented the same as in (Hauschild et al., [2022](https://doi.org/10.1093/bioinformatics/btac065)), where each client trains a local random forest, and the server aggregates them by combining trees from each client into a global forest. In our setup, each client trains a forest with 100 trees, and the server aggregates them into a global model with 500 trees since there are 5 clients in the federation. 

The shallow learner is a shallow adaptation of Federated Deep Regression Forests (Nolte et al., [2023](https://doi.org/10.1093/bioadv/vbad036)), except the feature extractor in our setup is a shallow multi-layer perceptron (MLP) with only 1 hidden layer. The shallow regression forest consists of an input layer, a hidden layer for feature extraction, and a probabilistic regression forest for prediction. Regression forests use the outputs of the feature extractor as probabilities for the nodes in a forest. In our setup, the feature extractor consists of 2 fully connected layers including the input layer with ReLu activation, batch normalization, and neuron sizes 672 to 128 to 128. The forest consists of 6 trees each with a depth of 4. 

For the DNN we used a 6-layer deep feed-forward neural network which was optimized for the full sample size NCI-60 dataset as described in (Nolte et al., [2023](https://doi.org/10.1093/bioadv/vbad036)). The 6 layers contain neuron sizes of 672 to 1000 to 800 to 500 to 200 to 100 to 1 with batch normalization in between each layer and ReLu activation. 

## Package Usage
To train and test the models, run the runSim.py file with the desired input parameters. For example, the following 2 lines show examples of how to train a Federated DRF model and subsequently test it.
```
python runSim.py -trainFed -model FedDRF
python runSim.py -evalFed -model FedDRF
```
While training, the progress will be printed and the model will be saved in the save directory specified with the save_dir parameter. The following train validation and test results are the expected output of the evaluation with default parameters:
```
FedDRF
Train
0.4508116880631354
0.6665665238011472
Val
0.7059264871384447
0.40195852183943825
Test
0.7155838751232734
0.37935022459737705
```
### Main Parameters
```
-data_path    Path to data directory: Default '../data'
-dataset      Dataset directory name: Default 'NCI60'
-save_dir     Directory location for saving models: Default '../models/'
-save_name    Name for saving models: Default 'model'
-seed         Seed for controlling random data partition: Default 2020
-beta         Beta of the Dirichlet distribution that controls the
              heterogeneity of the federated dataset across clients: Default None
-numSamples   number of samples to include in the federation: Default 1000
-model        Choose from ['FedDRF', 'FedRF', 'ANNFull']: Default 'FedDRF'
-trainCent    Boolean to train centralized alt: Default False
-trainFed     Boolean to train Federated model: Default True
-evalCent     Boolean to evaluate centralized alt: Default False
-evalFed      Boolean to evaluate Federated model: Default False
```
### Other Parameters
```
-n_tree: Default 6
-tree_depth: Default 4
-num_output: Default 128
-batch_size: Default 128 (Gets changed based on dynamic batch size set based on federation sample size)
-eval_batch_size: Default 128 
-leaf_batch_size: Default 128  
-label_batch_size: Default 40000 (use all samples when updating)  
-label_iter_time: Default 3  
-epochs: Default 200 (Gets stopped before with early stopping and learning rate reduction)
-lr: Default 1e-3  (initial learning rate)
-clientlr: Default 1e-3  (initial client learning rate)
-ESpat: Default 10  (Early stopping patience)
-LRsched: Default 5  (Learning rate reduction patience)
```
## Varying number of samples available to the federation 

### Simulation Setup
The federation consisted of 5 clients and a coordinating server. The data was split into training, validation, and test datasets. The number of training samples varied from [100,250,500,1000,2500,5000,10000,20000,40000], while 2000 samples were used for validation data at the server and the remaining samples were used as test data. The training data was evenly split among the 5 clients while the validation and test data were kept with the server for early stopping and learning rate reduction as well as evaluation of the models. The same data partitions were used across the 3 models and the experiments were run for 5 random partitions/seeds and the results were averaged.
The federated random forest was trained with one communication round while the shallow learner and DNN were trained with Federated Averaging over multiple communication rounds. The shallow learner and DNN used the validation data for early stopping and learning rate reduction with patience 10 and 5 respectively. The initial learning rate used for both federated averaging models was 1e-3 with the Adam optimizer. We employed an adaptive batch size to improve fairness by avoiding local client progress heterogeneity (Li et al., [2022](https://ieeexplore.ieee.org/abstract/document/9835537)) across sample sizes for the neural network-based models. The batch size was set to keep the number of local training steps for each client consistent at 100 per epoch across sample size, i.e., batch size = # Samples/100. The batch size was then clipped to the range [8, 256] to avoid the extreme low and high batch size cases, which would impact the batch normalization and memory requirements respectively. 

### Results
![Sample Size Results](https://github.com/DanielNolte/FLSimulation/blob/main/SampleSizeAnalysis.png)
The above figure shows the MSE for the 3 models with varying sample sizes. The results clearly show the importance of traditional machine learning and shallow learning in federated scenarios with limited samples as compared to deep learning. As expected, the federated random forest achieves the lowest MSE on the small sample cases with 100-500 samples. The shallow regression forest performs the best on the 1000-10000 sample cases and the DNN performed the best on the high sample cases of 10000-40000 samples. This highlights the significance of traditional machine learning methods in federated settings with limited samples. These results are generated with the default parameters except for varying the numSamples parameter with [100,250,500,1000,2500,5000,10000,20000,40000] and varying the seed with [2020,2021,2022,2023,2024], for each of the models ['FedDRF', 'FedRF', 'ANNFull'] and averaging each across the 5 random partitions.

## Varying degree of client heterogeneity across the federation

### Simulation Setup
To induce client heterogeneity, we quantized the overall federated samples based on their target value into 5 bins of equal size, matching the number of clients in the federation. We then treated the problem as a label distribution skew simulation and followed the common practice of sampling client bin percentages from a Dirichlet distribution (Zhao et al., [2018](https://doi.org/10.48550/arXiv.1806.00582)). The probabilities were sampled from Dir(β), with β being a vector of a repeating constant with length equal to the number of clients. This results in a square 5x5 matrix with rows summing to 1, each representing a bin and its elements representing the percent of samples each client receives from it. The columns, which represent clients, do not sum to one, so this method also induces client sample size skew in addition to the client distribution skew. As the beta constant gets smaller, the percentages have more variance leading to more heterogenous clients. The beta values used were [10, 2.5, 1, and 0.1], which equate to homogenous, mild heterogeneity, moderate heterogeneity, and severe heterogeneity. When sampling the client percentages, we verify that each client receives at least 5% of the samples to ensure that each client receives enough samples in the severe heterogeneity cases. We varied the number of training samples from [500, 1000, 10000] to allow a fair view for each level of model complexity and ran each case for 5 different partitions/seeds of the A549 cell line. 
### Results
![Heterogeneity Results](https://github.com/DanielNolte/FLSimulation/blob/main/HeterogeneityAnalysis.png)
The above figure depicts the MSE and Pearson correlation coefficient (PCC) of the competing models with varying degrees of heterogeneity and number of samples in the federation. The figure shows that all models are negatively affected by client heterogeneity and that the best model is typically the one with complexity optimized for the given sample size. Additionally, Federated Random Forest maintains a more stable PCC compared to the other deep learning methods. The deep learning methods have more hyper-parameters that require tuning as they can greatly influence the iterative learning process, such as a balance between client learning rate, batch size, and number of local client updates per communication round (Li et al., [2022](https://ieeexplore.ieee.org/abstract/document/9835537)). An imbalance of these parameters under heterogeneity can cause local progress heterogeneity which can severely impact the learning process but can be mitigated by a more extensive aggregation technique FedNova (Wang et al., [2020](https://proceedings.neurips.cc/paper/2020/file/564127c03caab942e503ee6f810f54fd-Paper.pdf)). Whereas the federated random forests were trained using only 1 communication round and minimal hyper-parameters, leading to a more efficient and timely learning process. These results are generated with the default parameters except for varying the numSamples parameter with [500, 1000, 10000], varying beta with [10, 2.5, 1, 0.1], and varying the seed with [2020,2021,2022,2023,2024], for each of the models ['FedDRF', 'FedRF', 'ANNFull'] and averaging each across the 5 random partitions.
