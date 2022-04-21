----------------------------------------------------------------
BASE INFO:
----------------------------------------------------------------

validation accuracy is 81%, with only 1 epoch(no resourses for testing).(increases epochs  will easily reach target).
statification is used in sampling training and validation datasets.
Last_classification_layer of vgg19_nb has been modified according to task needs.

tested common vision models: vgg_family, resnet_family, resnext_family...
vgg19_nb gave the best/fastest results  with batch_normalization, at training batche_size=64.
Traning and valication loss/accuracy  is printed in training process...


----------------------------------------------------------------
USAGE:
----------------------------------------------------------------
For training script to run the following terminal (tested on windows) command is needed:

model_build.py -data_dir="full path to extracted data" -model_save_dir="folder path to model saving direcory" -n_epochs="number of epochs"

-data_dir="full path to extracted data" 
-model_save_dir="folder path to model saving direcory" 
-n_epochs="number of epochs"


