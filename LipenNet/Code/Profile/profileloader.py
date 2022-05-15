import Code.Protocol.errorcodes as err
import os.path
import sys
from typing import TypedDict,  get_type_hints

import Code.Protocol.enums as en


# noinspection PyTypedDict
class Hparams(TypedDict):
    #Dataset Parameters (paths & names)
    data_dir : str | None                               #Data directory name (folder in which are datasets):
    dataset_dir : str | None                            #Chosen Dataset Directory
    dataset_name : en.DatasetName | None                #Dataset name (must be registered by enum DatasetName)
    #Sub sets directory names
    trainset_dir : str | None                           #Names of dirs where are train set,
    valset_dir : str | None                             # eval set
    testset_dir : str | None                            # and testset , (if named "test" set as "test")
    label_filename : str | None                         #Name of the csv file with the labels
    train_batch_size :int |None
    val_batch_size: int | None
    test_batch_size: int | None

    #Training Parameters
    train_initial_learning_rate: float | None
    train_scheduler_list: list[int] | None
    train_single_batch_test: bool | None
    train_max_epoch: int | None
    train_device: en.Device | None
    train_model: en.ModelType | None
    train_optimizer : en.OptimizerType | None
    train_criterion : en.CriterionType | None



    #Normalizaton Parameters
    clean_dataset_mean : list[float] | None
    clean_dataset_std  : list[float] | None

    unified_dataset_mean: list[float] | None
    unified_dataset_std: list[float] | None

    merged_dataset_mean: list[float] | None
    merged_dataset_std:  list[float] | None


    #Augmentation Parameters
    augmentation_type : en.AugmentationType | None
    resize_size : tuple[int,int] | None
    vertical_flip_prob : float | None
    horizontal_flip_prob: float | None

    color_jitter_prob: float | None
    color_jitter_brightness: tuple[float,float] | None
    color_jitter_contrast: tuple[float,float] | None
    color_jitter_saturation: tuple[float,float] | None
    color_jitter_hue: tuple[float,float] | None

    gaussian_noise_prob: float | None
    gaussian_noise_mean: float | None
    gaussian_noise_std: float | None
    gaussian_blur_prob: float | None
    gaussian_blur_kernel_size: tuple[int, int] | None
    gaussian_blur_sigma : tuple[float, float] | None

    enhance_brightness_prob: float | None
    enhance_brightness_brightness_intensity: float | None
    enhance_brightness_max_brightness: float | None

    random_invert_prob: float | None
    random_equalize_prob: float | None
    random_greyscale_prob: float | None
    random_rotation_prob: float | None
    random_rotation_degrees: tuple[int, int] | None

    #Ratate Augmentation params
    rotate_angles : list[int] | None


    def fun(self,key,value):
        self[key] = value


def convertStrToType(key,value):
    set_type = Hparams.__annotations__[key].__args__[0]
    if set_type is None:
        print("None type value in params detected!")
        exit(err.PROFILE_NONE_VALUE_PARAM)
    elif set_type == str:
        return str(value)
    elif set_type == int:
        return int(value)
    elif set_type == float:
        return float(value)
    elif set_type == bool:
        return bool(value)
    elif set_type == list[float]:
        return list(map(float,(value[1:-1].split(","))))
    elif set_type == list[int]:
        return list(map(int,(value[1:-1].split(","))))
    elif set_type == tuple[float] or set_type == tuple[float,float] :
        return tuple(map(float, (value[1:-1].split(","))))
    elif set_type == tuple[int] or set_type == tuple[int,int] :
        return tuple(map(int, (value[1:-1].split(","))))
    elif set_type == en.DatasetName:
        return en.DatasetName[value]
    elif set_type == en.AugmentationType:
        return en.AugmentationType[value]
    elif set_type == en.Device:
        return en.Device[value]
    elif set_type == en.ModelType:
        return en.ModelType[value]
    elif set_type == en.OptimizerType:
        return en.OptimizerType[value]
    elif set_type == en.CriterionType:
        return en.CriterionType[value]

    else:
        print("Unimplemented Type Detected! -> " + str(set_type))
        sys.exit(err.PROFILE_UNIMPLEMENTED_TYPE)




#Hiperparameters of this program
__hparams  : Hparams = \
{
    #Data Paths:
    #Main Data Directory
    "data_dir" : None,
    #Main Dataset Directory
    "dataset_dir" : None,
    "dataset_name" : None,
    "trainset_dir" : None,
    "valset_dir" : None,
    "testset_dir" : None,
    "label_filename" : None,
    "train_batch_size" :None,
    "val_batch_size":  None,
    "test_batch_size":  None,

    #Training Params
    "train_initial_learning_rate" : None,
    "train_scheduler_list" : None,
    "train_single_batch_test": None,
    "train_max_epoch":None,
    "train_device" :None,
    "train_model": None,
    "train_optimizer": None,
    "train_criterion": None,

    # Normalizaton Parameters
    "clean_dataset_mean": None,
    "clean_dataset_std":  None,

    "unified_dataset_mean":  None,
    "unified_dataset_std":   None,

    "merged_dataset_mean":   None,
    "merged_dataset_std":   None,

    # Augmentation Parameters
    "augmentation_type":  None,
    "resize_size":  None,
    "vertical_flip_prob":   None,
    "horizontal_flip_prob":   None,

    "color_jitter_prob":   None,
    "color_jitter_brightness":   None,
    "color_jitter_contrast":   None,
    "color_jitter_saturation":   None,
    "color_jitter_hue":   None,

    "gaussian_noise_prob":   None,
    "gaussian_noise_mean": None,
    "gaussian_noise_std": None,
    "gaussian_blur_prob":   None,
    "gaussian_blur_kernel_size": None,
    "gaussian_blur_sigma":   None,

    "enhance_brightness_prob":   None,
    "enhance_brightness_brightness_intensity":   None,
    "enhance_brightness_max_brightness":   None,

    "random_invert_prob":   None,
    "random_equalize_prob":   None,
    "random_greyscale_prob":   None,
    "random_rotation_prob":   None,
    "random_rotation_degrees":  None,

    # Ratate Augmentation params
    "rotate_angles":  None,
}


def loadProfile(arguments):
    if len(arguments) < 2:
        print("Pass the profile file name as an argument to script")
        sys.exit(err.PROFILE_WRONG_PROGRAM_ARG_NUM)
    profile_name = arguments[1]
    profile_path = "Code/Profile/Profiles/" + profile_name
    if not os.path.exists(profile_path):
        print("Wrong argument passed. Not a file. Pass the profile file name")
        sys.exit(err.PROFILE_WRONG_PARAM_NOT_FILE)
    with open("Code/Profile/Profiles/" + profile_name) as profile_file:
        for line in profile_file:
            if line[0] == "\n" : continue #ignore empty lines
            if line[0] == "#" : continue #ignore commands
            line = line.rstrip()
            line = ''.join(line.rsplit())
            if len(line.split("=")) < 2:
                print("Add value to the param in profile file (after = )")
                sys.exit(err.PROFILE_EMPTY_PARAM_VALUE)
            parameter_key : str = line.split("=")[0]
            parameter_value = line.split("=")[1]
            if parameter_key in __hparams:
                parameter_value = convertStrToType(parameter_key,parameter_value)
                __hparams[parameter_key] = parameter_value
            else:
                print("Wrong key in profile.txt -> "+ parameter_key)
                sys.exit(err.PROFILE_WRONG_KEY_IN_PROFILE_FILE)
        #Check if all hparams are set:
        if any( elem is None for elem in __hparams.values()):
            print("Not all parameters set!")
            sys.exit(err.PROFILE_NOT_ALL_PARAMS_SET)
        return __hparams


