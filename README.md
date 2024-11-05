# FedSal: Enhancing Federated Graph Learning Through Saliency Aware Client Clustering

This repository contains the implementation of the paper:

> [FedSal: Enhancing Federated Graph Learning Through Saliency Aware Client Clustering](#)

## Requirements

It is recommended to use a Conda environment with Python version **3.10.13** to ensure compatibility and ease of dependency management.

### Setting Up the Conda Environment

1. **Install Conda**: If you haven't installed Conda yet, download and install it from [here](https://docs.conda.io/en/latest/miniconda.html).

2. **Create a Conda Environment**:
   
   ```bash
   conda create -n fedsal_env python=3.10.13
   ```

3. **Activate the Environment**:
   
   ```bash
   conda activate fedsal_env
   ```

4. **Install PyTorch**:
   
   Ensure that you install the correct version of PyTorch compatible with your system. The following command installs PyTorch **1.12.0** with CUDA 10.2. If you don't require CUDA support, you can install the CPU-only version.

   ```bash
   # For CUDA 10.2
   conda install pytorch==1.12.0 torchvision torchaudio cudatoolkit=10.2 -c pytorch

   # For CPU-only
   # conda install pytorch==1.12.0 torchvision torchaudio cpuonly -c pytorch
   ```

5. **Install Other Requirements**:
   
   ```bash
   pip install -r requirements.txt
   ```

   *Note: Ensure that `requirements.txt` is present in the repository root.*

## Running the Project

### Run Once for a Specific Setting

#### 1. OneDS: Distributing One Dataset to Multiple Clients

```bash
python main_oneDS.py --repeat {repeat_index} --data_group {dataset} --num_clients {num_clients} --seed {random_seed} --epssal1 {epsilon_1} --epssal2 {epsilon_2}
```

**Parameters:**
- `--repeat`: Index of the repeat (e.g., 1, 2, 3, ...)
- `--data_group`: Name of the dataset (e.g., PROTEINS, IMDB-BINARY, NCI1)
- `--num_clients`: Number of clients (e.g., 10, 30)
- `--seed`: Random seed for reproducibility
- `--epssal1` and `--epssal2`: Epsilon values for saliency

#### 2. MultiDS: Multiple Datasets with Each Client Owning One Dataset

```bash
python main_multiDS.py --repeat {repeat_index} --data_group {datagroup} --seed {random_seed} --epssal1 {epsilon_1} --epssal2 {epsilon_2}
```

**Parameters:**
- `--repeat`: Index of the repeat
- `--data_group`: Pre-defined data group in `setupGC.py` (e.g., molecules, biochem, mix)
- `--seed`: Random seed
- `--epssal1` and `--epssal2`: Epsilon values for saliency

### Run Repetitions for All Datasets

#### 1. Execute All Repetitions

```bash
bash run_file
```

*Ensure that `run_file` has the appropriate execute permissions. You can set it using `chmod +x run_file`.*

#### 2. Aggregate and Generate Results

To average all repetitions and obtain overall performance metrics:

```bash
python GenerateResults.py
```

Alternatively, you can manually run all commands to recreate the results as shown below.

### Manual Execution of All Commands

#### Multi Dataset

```bash
# Molecules Dataset
python main_multiDS.py --repeat 1 --data_group molecules --seed 1 
python main_multiDS.py --repeat 2 --data_group molecules --seed 2 
python main_multiDS.py --repeat 3 --data_group molecules --seed 3 
python main_multiDS.py --repeat 4 --data_group molecules --seed 4 
python main_multiDS.py --repeat 5 --data_group molecules --seed 5 

# Biochem Dataset
python main_multiDS.py --repeat 1 --data_group biochem --seed 1 
python main_multiDS.py --repeat 2 --data_group biochem --seed 2 
python main_multiDS.py --repeat 3 --data_group biochem --seed 3 
python main_multiDS.py --repeat 4 --data_group biochem --seed 4 
python main_multiDS.py --repeat 5 --data_group biochem --seed 5 

# Mix Dataset
python main_multiDS.py --repeat 1 --data_group mix --seed 1 
python main_multiDS.py --repeat 2 --data_group mix --seed 2 
python main_multiDS.py --repeat 3 --data_group mix --seed 3 
python main_multiDS.py --repeat 4 --data_group mix --seed 4 
python main_multiDS.py --repeat 5 --data_group mix --seed 5 
```

#### Single Dataset

```bash
# PROTEINS Dataset with 10 Clients
python main_oneDS.py --repeat 1 --data_group PROTEINS --num_clients 10 --seed 1
python main_oneDS.py --repeat 2 --data_group PROTEINS --num_clients 10 --seed 2
python main_oneDS.py --repeat 3 --data_group PROTEINS --num_clients 10 --seed 3
python main_oneDS.py --repeat 4 --data_group PROTEINS --num_clients 10 --seed 4
python main_oneDS.py --repeat 5 --data_group PROTEINS --num_clients 10 --seed 5

# IMDB-BINARY Dataset with 10 Clients
python main_oneDS.py --repeat 1 --data_group IMDB-BINARY --num_clients 10 --seed 1
python main_oneDS.py --repeat 2 --data_group IMDB-BINARY --num_clients 10 --seed 2
python main_oneDS.py --repeat 3 --data_group IMDB-BINARY --num_clients 10 --seed 3
python main_oneDS.py --repeat 4 --data_group IMDB-BINARY --num_clients 10 --seed 4
python main_oneDS.py --repeat 5 --data_group IMDB-BINARY --num_clients 10 --seed 5

# NCI1 Dataset with 30 Clients
python main_oneDS.py --repeat 1 --data_group NCI1 --num_clients 30 --seed 1
python main_oneDS.py --repeat 2 --data_group NCI1 --num_clients 30 --seed 2
python main_oneDS.py --repeat 3 --data_group NCI1 --num_clients 30 --seed 3
python main_oneDS.py --repeat 4 --data_group NCI1 --num_clients 30 --seed 4
python main_oneDS.py --repeat 5 --data_group NCI1 --num_clients 30 --seed 5
```

### Outputs

The repetition results will be stored in the following directories:

- **OneDS Setting**:
  ```
  ./outputs/oneDS-nonOverlap/{dataset}-{numClients}clients/repeats/
  ```

- **MultiDS Setting**:
  ```
  ./outputs/multiDS-nonOverlap/{datagroup}/repeats/
  ```

After running `GenerateResults.py`, the aggregated results will be printed to the console and saved in:

```
./final_output
```

*Note: Various arguments can be defined for different settings. If the arguments `datapath` and `outbase` are not specified, datasets will be downloaded to `./data`, and outputs will be stored in `./outputs` by default.*

## Citation

If you find this project helpful, please consider citing the following paper:

```bibtex

```