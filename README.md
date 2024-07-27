# FedSal: Enhancing Federated Graph Learning Through Saliency Aware Client Clustering
 

This repository contains the implementation of the paper:

> [FedSal: Enhancing Federated Graph Learning Through Saliency Aware Client Clustering]()

## Requirements

To install requirements:

```setup
pip3 install torch==1.12.0
pip3 install -r requirements.txt
```

## Run once for one certain setting

(1) OneDS: Distributing one dataset to a number of clients:

```
python main_oneDS.py --repeat {index of the repeat} --data_group {dataset} --num_clients {num of clients} --seed {random seed}  --epssal1 {epsilon_1} --epssal2 {epsilon_2}
```

(2) MultiDS: For multiple datasets, each client owns one dataset (datagroups are pre-defined in ___setupGC.py___):

```
python main_multiDS.py --repeat {index of the repeat} --data_group {datagroup} --seed {random seed} --epssal1 {epsilon_1} --epssal2 {epsilon_2} 
```


## Run repetitions for all datasets
(1) To get all repetition results:

```
bash run_file
```
(2) To averagely aggregate all repetitions, and get the overall performance:

```
python GenerateResults.py
```

Or, to run all commands manually to recreate results:

```
# multi dataset
python main_multiDS.py --repeat 1 --data_group molecules --seed 1 
python main_multiDS.py --repeat 2 --data_group molecules --seed 2 
python main_multiDS.py --repeat 3 --data_group molecules --seed 3 
python main_multiDS.py --repeat 4 --data_group molecules --seed 4 
python main_multiDS.py --repeat 5 --data_group molecules --seed 5 

python main_multiDS.py --repeat 1 --data_group biochem --seed 1 
python main_multiDS.py --repeat 2 --data_group biochem --seed 2 
python main_multiDS.py --repeat 3 --data_group biochem --seed 3 
python main_multiDS.py --repeat 4 --data_group biochem --seed 4 
python main_multiDS.py --repeat 5 --data_group biochem --seed 5 

python main_multiDS.py --repeat 1 --data_group mix --seed 1 
python main_multiDS.py --repeat 2 --data_group mix --seed 2 
python main_multiDS.py --repeat 3 --data_group mix --seed 3 
python main_multiDS.py --repeat 4 --data_group mix --seed 4 
python main_multiDS.py --repeat 5 --data_group mix --seed 5 

# single dataset
python main_oneDS.py --repeat 1 --data_group PROTEINS --num_clients 10 --seed 1
python main_oneDS.py --repeat 2 --data_group PROTEINS --num_clients 10 --seed 2
python main_oneDS.py --repeat 3 --data_group PROTEINS --num_clients 10 --seed 3
python main_oneDS.py --repeat 4 --data_group PROTEINS --num_clients 10 --seed 4
python main_oneDS.py --repeat 5 --data_group PROTEINS --num_clients 10 --seed 5

python main_oneDS.py --repeat 1 --data_group IMDB-BINARY --num_clients 10 --seed 1
python main_oneDS.py --repeat 2 --data_group IMDB-BINARY --num_clients 10 --seed 2
python main_oneDS.py --repeat 3 --data_group IMDB-BINARY --num_clients 10 --seed 3
python main_oneDS.py --repeat 4 --data_group IMDB-BINARY --num_clients 10 --seed 4
python main_oneDS.py --repeat 5 --data_group IMDB-BINARY --num_clients 10 --seed 5

python main_oneDS.py --repeat 1 --data_group NCI1 --num_clients 30 --seed 1
python main_oneDS.py --repeat 2 --data_group NCI1 --num_clients 30 --seed 2
python main_oneDS.py --repeat 3 --data_group NCI1 --num_clients 30 --seed 3
python main_oneDS.py --repeat 4 --data_group NCI1 --num_clients 30 --seed 4
python main_oneDS.py --repeat 5 --data_group NCI1 --num_clients 30 --seed 5
```

### Outputs
The repetition results started with '{\d}_' will be stored in:
> _./outputs/oneDS-nonOverlap/{dataset}-{numClients}clients/repeats/_, for the OneDS setting;

> _./outputs/multiDS-nonOverlap/{datagroup}/repeats/_, for the MultiDS setting.

After running Generating Results, the results are printe and saved files are stored in folder
> _./final_output


*Note: There are various arguments can be defined for different settings. If the arguments 'datapath' and 'outbase' are not specified, datasets will be downloaded in './data', and outputs will be stored in './outputs' by default.


## Citation
If you find this project helpful, please consider to cite the following paper:
```

```