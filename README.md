# ARC
Autonomous Learning Rate Control (ARC) Guide.

## Pre-requisites

* TensorFlow == 2.4.1
* FastEstimator == 1.2

## Data
Data used to train ARC model is stored in `data/offline_data.pkl`, it contains 5050 samples of training loss (resized), valdiation loss, and learning rate history from running 12 different tasks multiple times. Check Table 1 of the paper for detail.

offline_data.pkl is the pickle file storing the dictionary with following keys:
* "train_loss": 5050 training loss history with [100, 200, 300] length
* "val_loss": 5050 validation loss history with 1~30 length
* "train_lr": 5050 learning rate history with [1, 2, 3] length

  (length of train_lr elements will be the length of train_loss divided by 100)

* "label": 5050 correct actions to take. each element is integer either 0, 1, or 2.

  0: raise the learning rate

  1: keep the learning constant

  2: decrease the learning rate

Data for testing (Cifar10, PTB and MSCOCO) will be downloaded automatically to your specified folder when you perform testing for the first time.

## ARC model
The trained weights used in the paper is in model/arc.h5. If you would like to retrain the ARC, simply go to the model folder, then
```
fastestimator train train_arc.py
```
This command will train ARC and store the trained model in `model/checkpoint`

## ARC model selection (optional)
Due to the high performace variance of ARC, we use a proxy task to select the final ARC model. The proxy task is an image classification training of WideResNet28 on SVHN_Cropped with high (0.1), medium(0.001), low(0.00001) learning rate. By running the proxy task 5 times for each learning rate (totally 5*3 times) and average the maximum validation accuracy you can get the proxy score of the ARC model. The `model/arc.h5` was selected from 10 independent candidates.

to run single proxy task:
1. dowload SVHN_Cropped datasets and re-organize the datasets as the following file structure <br>
train: http://ufldl.stanford.edu/housenumbers/train_32x32.mat <br>
test: http://ufldl.stanford.edu/housenumbers/test_32x32.mat
```
    SVHN_Cropped
    ├── train/
    │   ├── 0/
    │   |   ├── xxx.png
    |   |   └── ...
    |   ...
    │   └── 9/
    │       ├── xxx.png
    |       └── ...
    └── test/
        ├── 0/
        |   ├── xxx.png
        |   └── ...
        ...
        └── 9/
            ├── xxx.png
            ├── xxx.png
            └── ...

```

2. run the command
```
fastestimator source/proxy_test/wideresnet_svhn.py --data_dir <path_to_SVHN_Cropped> --init <init_lr> --weight_path model/arc.h5
```


## Testing the ARC
The testing scripts used in experiments are stored in `source` folder. There are not only ARC testing but also some LR scheduler for comparision.

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
