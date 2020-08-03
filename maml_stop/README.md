# opts_metal
meta learning with optimal stopping


## Setup

Navigate to the root of this meta learning subfolder, and do

    pip install -e .


## Omniglot task-imbalaced few-shot learning

Firstly, we split the data using the exact same split as MAML tensorflow code (https://github.com/cbfinn/maml). Note that the corresponding split indices are already hard-coded in this repo. 

    cd metal/experiments/omniglot
    python make_split.py

After that, one can simply run the script (the first run would download the dataset automatically)

    ./run_imba.sh

You can also configure the parameteres in the script to try out different options. After the default number of updates, we can test the performance with additional arguments:

    ./run_imba.sh -phase test -epoch_load 600

The above command would test against the 'test' split with 600 tasks by default. The model dump it loads is after 60k batchs of training.

## MiniImagenet task-imbalaced few-shot learning

We also provide the script under `metal/experiments/miniimagenet`. 
The process is almost the same as above Omniglot experiment, except that no explicit data split preparation is needed.

## Vanilla Few-shot Learning

We also provide the scripts for vanilla (balanced) few-shot learning, under each experiment folder. The training/test usage of the scripts are similar to above.