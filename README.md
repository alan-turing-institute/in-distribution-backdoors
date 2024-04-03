# IEEE Security and Privacy Workshop (DLSP) 2024: Mitigating Deep Reinforcement Learning Backdoors in the Neural Activation Space
## Atari Breakout Experiments: Section 4

This repository contains the source code of sanitization backdoor policies for Atari breakout game environment. The backdoor policy in this example has been trained using the environment poisoning framework of TrojDRL [paper](https://arxiv.org/pdf/1903.06638.pdf) .

The state space consists of a concatenated image frames. The trigger is a 3x5 image inserted on the tile space of the Atari Breakout Game. The backdoor policy has been trained to a level so that in absense of trigger the policy consistently achieves high score against the oppenent while in presence of trigger it takes 'no move' action eventually achieving a very low score on average.


### Setup codebase and python environment.

1. install anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/).
2. create a new environment from the specification file.
 ```conda env create --name NEW_ENV_NAME -f environment.yml```
3. activate conda environment.
 ```conda activate NEW_ENV_NAME```

### Run the code. 
1. test backdoor policy in the clean environment :  
	 ```python driver_parallel.py 'backdoor_in_clean' 'save_states'```
	- change number of trials, number of test episodes(test_count) in the trials if needed.
	- the clean states data generated here would be used for sanitization in step 3.
2. test backdoor policy in the triggered environment :  
	 ```python driver_parallel.py 'backdoor_in_triggered'```
3. sanitize backdoor and test sanitized policy in the triggered environment :  
	```python driver_parallel.py 'sanitized_in_triggered'```
	- construct sanitized policies for various number of clean sample sets and then test it.
4. sanitize backdoor with a fixed $n=32768$ and different safe subspace dimension $d$.
     ```python driver_parallel.py 'sanitized_with_fixed_n'```
	- to run this part, we need to have bases for $n=32768$ samples obtained from step 3. 



### Training the backdoor policy from scratch.
- We train a strongly targeted backdoor policy that uses a  and takes 'no move' action when the trigger is active as specfied in the TrojDRL paper. For more details please refer to this paper and the code.
- To train this backdoor policy run :
```
python3 train.py --game=breakout --debugging_folder=pretrained_backdoor/strong_targeted/breakout_target_noop/ --poison --color=5 --attack_method=strong_targeted --pixels_to_poison_h=5 --pixels_to_poison_v=3 --start_position="29,28" --when_to_poison="uniformly" --action=2 --budget=20000 --device='/cpu:0' --emulator_counts=12 --emulator_workers=4
```
### In-distribution trigger

[triggered_state.png](https://github.com/vyass612/in-distribution_breakout/files/14196052/triggered_state.png)


### Results

Our results show that our in-distribution trigger successfully evades the defence algorihtm of Bharti et al's NeurIPS solution [paper](https://openreview.net/forum?id=11WmFbrIt26)

[performance_breakout.pdf](https://github.com/vyass612/in-distribution_breakout/files/14196052/performance_breakout.pdf)

[spectrum_safe_subspace.pdf](https://github.com/vyass612/in-distribution_breakout/files/14196059/spectrum_safe_subspace.pdf)


### Edited Files 

The ```evaluator.py ``` file contains the code which changes the size of the trigger along with the ```params_indist.yml``` file. The latter file adjusts the default size along with the colour of the trigger
The ```plot_graphs.py``` file saves the visualisation found in figure 2 of the paper, whilst the ```analyse_performance_for_n=32768_sanitization.py``` file saves the visualisation found in figure 3 of the paper

## Minigrid Experiments: Section 5

### Running this code (ran on a Mac)

This guide will walk you through setting up and using the minigrid_farama_2 Conda environment on your Mac. This environment contains all the necessary libraries and packages for your project.

### Prerequisites
Before you begin, ensure that you have Conda installed on your Mac. If you do not have Conda installed, please follow the [official Conda installation guide for a Mac](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 

### Installation 

Clone the Repository

Create the conda environment
```python
conda env create -f environment.yml
conda activate minigrid_farama_2
```
Set PYTHONPATH
```python
export PYTHONPATH= "/path/to/file/Minigrid"
echo $PYTHONPATH
```

### Execution

Run the visualize file, and edit the "visualize.py" file and "crossings.py" code according to the data you want to collect (Non-triggered/Triggered, Goal in field of view, Trigger in field of view, Thresholding Detector Algorithm)(Trigger on/Non-Trigger off). The visualisation file collects the neural activations for every step and saves them to a file according to the type of data that requires collection,
```python
python3 -m scripts.visualize --env MiniGrid-LavaCrossingS9N1-v0 --model DSLP_Crossings_Trigger_60k_256_neurons --episodes 1000
```

To run the training file from scratch, and edit the "visualize.py" file and "crossings.py" code according to the data you want to collect (Non-triggered/Triggered, Goal in field of view, Trigger in field of view, Thresholding Detector Algorithm)(Trigger on/Non-Trigger off). The train.py file will save all model outputs to the ```Minigrid/minigrid/torch-ac/rl-starter-files/storage``` folder. This model can then be accessed in the visualize.py file above 
```python
python3 -m scripts.train --algo ppo --env MiniGrid-LavaCrossingS9N1-v0  --model model_name --save-interval 10 --frames 60000000
```




