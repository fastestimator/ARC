# ARC
Autonomous Learning Rate Control (ARC) Guide.

## Pre-requisites

* TensorFlow == 2.4.1
* FastEstimator == 1.2

## Data
Data used to train ARC model is stored in data/offline_data.pkl, it contains 5050 samples of training loss (resized), valdiation loss, and learning rate history from running 12 different tasks multiple times. Check Table 1 of the paper for detail.

Data for testing (Cifar10, PTB and MSCOCO) will be downloaded automatically to your specified folder when you perform testing for the first time.

## ARC model
The trained weights used in the paper is in model/arc.h5. If you would like to retrain the ARC, simply go to the model folder, then
```
fastestimator train train_arc.py
```
This command will train ARC and stored the model in model/checkpoint 

## Testing the ARC
The testing scripts used in experiments are stored in `test` folder, to run any particular experiment, simply go to the corresponding folder.

### Run Cifar10 tests:
* Base LR:
```
fastestimator train source/normal_compare/image_classification/base_lr.py --init_lr 1e-2
```
* Cosine Decay:
```
fastestimator train source/normal_compare/image_classification/cosine_decay.py --init_lr 1e-2
```
* Cyclic Cosine Decay:
```
fastestimator train source/normal_compare/image_classification/cyclic_cosine_decay.py --init_lr 1e-2
```
* Exponential Decay:
```
fastestimator train source/normal_compare/image_classification/exponential_decay.py --init_lr 1e-2
```
* ARC:
```
fastestimator train source/normal_compare/image_classification/arc.py --weights_path model/arc.h5 --init_lr 1e-2
```

### Run PTB tests:
* Base LR:
```
fastestimator train source/normal_compare/language_modeling/base_lr.py --init_lr 1.0 --data_dir /folder/to/download/data
```
* Cosine Decay:
```
fastestimator train source/normal_compare/language_modeling/cosine_decay.py --init_lr 1.0 --data_dir /folder/to/download/data
```
* Cyclic Cosine Decay:
```
fastestimator train source/normal_compare/language_modeling/cyclic_cosine_decay.py --init_lr 1.0 --data_dir /folder/to/download/data
```
* Exponential Decay:
```
fastestimator train source/normal_compare/language_modeling/exponential_decay.py --init_lr 1.0 --data_dir /folder/to/download/data
```
* ARC:
```
fastestimator train source/normal_compare/language_modeling/arc.py --weights_path model/arc.h5 --init_lr 1.0 --data_dir /folder/to/download/data
```

### Run MSCOCO tests:
* Base LR:
```
fastestimator train source/normal_compare/instance_detection/base_lr.py --init_lr 1e-2 --data_dir /folder/to/download/data
```
* Cosine Decay:
```
fastestimator train source/normal_compare/instance_detection/cosine_decay.py --init_lr 1.0 --data_dir /folder/to/download/data
```
* Cyclic Cosine Decay:
```
fastestimator train source/normal_compare/instance_detection/cyclic_cosine_decay.py --init_lr 1e-2 --data_dir /folder/to/download/data
```
* Exponential Decay:
```
fastestimator train source/normal_compare/instance_detection/exponential_decay.py --init_lr 1e-2 --data_dir /folder/to/download/data
```
* ARC:
```
fastestimator train source/normal_compare/instance_detection/arc.py --weights_path model/arc.h5 --init_lr 1e-2 --data_dir /folder/to/download/data
```
