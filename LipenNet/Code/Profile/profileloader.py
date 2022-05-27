import Code.Protocol.errorcodes as err
import os.path
import sys
from typing import TypedDict,  get_type_hints

import Code.Protocol.enums as en

DEFAULT_PROFILE_DIR_PATH = "Code/Profile/Profiles/"


# noinspection PyTypedDict
class Hparams(TypedDict):
    #HParams file (saved form input argument)
    profile_file : str | None                           #Filled automatically
    #Dataset Parameters (paths & names)
    data_dir : str | None                               #Data directory name (folder in which are datasets):
    dataset_dir : str | None                            #Chosen Dataset Directory
    dataset_name : en.DatasetName | None                #Dataset name (must be registered by enum DatasetName)
    #Sub sets directory names
    trainset_dir : str | None                           #Names of dirs where are train set,
    valset_dir : str | None                             # eval set
    testset_dir : str | None                            # and testset , (if named "test" set as "test")
    label_filename : str | None                         #Name of the csv file with the labels
    normalization_filename : str | None                 #Name of the pickle file with normalization info
    train_batch_size :int |None
    val_batch_size: int | None
    test_batch_size: int | None

    #Load Params
    load_model : bool| None                             #Filled automatically (pass loadable (with path) model as run param)
    load_model_path : str | None                        #Filled automatically
    save_dir_path : str | None                          #Directory where models are saved (must be set manually)
    save_mode : en.SavingMode | None                    #defines whether to save model during (or after) training

    #Training Parameters
    initial_learning_rate: float | None
    weight_decay: float | None
    frozen_initial_layers: int | None
    scheduler_list: list[int] | None
    scheduler_gamma : float | None
    grad_per_batch : int | None
    single_batch_test: bool | None
    max_epoch: int | None
    train_device: en.Device | None
    model: en.ModelType | None
    optimizer : en.OptimizerType | None
    criterion : en.CriterionType | None
    reduction_mode : en.ReductionMode | None

    clean_class_weights : list[float] | None
    unified_class_weights: list[float] | None
    unaug_class_weights: list[float] | None
    merged_class_weights: list[float] | None
    meraug_class_weights: list[float] | None
    uniform20_class_weights: list[float] | None

    #Eval Parameters
    val_device: en.Device | None
    epoch_per_eval : int | None
    val_criterion : en.CriterionType | None

    #Augmentation Parameters
    augmentation_type : en.AugmentationType | None
    augmentation_count : int | None
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
        return value == "True"
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
    elif set_type == en.SavingMode:
        return en.SavingMode[value]
    elif set_type == en.ReductionMode:
        return en.ReductionMode[value]

    else:
        print("Unimplemented Type Detected! -> " + str(set_type))
        sys.exit(err.PROFILE_UNIMPLEMENTED_TYPE)




#Hiperparameters of this program
__hparams  : Hparams = \
{
    # HParams file
    "profile_file" : None,
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
    "normalization_filename" : None,
    "train_batch_size" :None,
    "val_batch_size":  None,
    "test_batch_size":  None,
    #Load Params
    "load_model" : None,
    "load_model_path" : None,
    "save_dir_path" : None,
    "save_mode": None,

    #Training Params
    "initial_learning_rate" : None,
    "weight_decay": None,
    "frozen_initial_layers": None,
    "scheduler_list" : None,
    "scheduler_gamma" : None,
    "grad_per_batch" : None,
    "single_batch_test": None,
    "max_epoch":None,
    "train_device" :None,
    "model": None,
    "optimizer": None,
    "criterion": None,
    "reduction_mode": None,
    "clean_class_weights": None,
    "unified_class_weights": None,
    "unaug_class_weights": None,
    "merged_class_weights": None,
    "meraug_class_weights": None,
    "uniform20_class_weights":None,

    #Eval Parameters
    "val_device":  None,
    "epoch_per_eval": None,
    "val_criterion": None,

    # Augmentation Parameters
    "augmentation_type":  None,
    "augmentation_count": None,
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
    """
    Function meant to load and decipher profile.txt file with hyper parameters
    :param arguments - run arguments passed from user ( arg 1 - profile.txt, arg 2 - loadable model path (optional )):
    :return - hparams dict with all hyper parameters set:
    """
    if len(arguments) < 2:
        print("Pass the profile file name as an argument to script")
        sys.exit(err.PROFILE_WRONG_PROGRAM_ARG_NUM)
    #decode 1st run argument  - profile file path or name
    profile_name : str = arguments[1]
    #decode 2nd (if present) run argument - optional - loadable model path
    load_model_path : str = arguments [2] if len(arguments) >= 3 else ""
    #profile path is default (DEFAULT_PROFILE_DIR_PATH + profile_name) or present in profile_name (if name contains any "/")
    profile_path = DEFAULT_PROFILE_DIR_PATH + profile_name if len(profile_name.split("/")) == 1 else profile_name
    if not os.path.exists(profile_path):
        print("Wrong argument passed. " + profile_path+ " Not a file. Pass the profile file name")
        sys.exit(err.PROFILE_WRONG_PARAM_NOT_FILE)
    with open(profile_path) as profile_file:
        #Check every line in profile file
        for line in profile_file:
            #ignore empty lines
            if line[0] == "\n" : continue
            #ignore comments
            if line[0] == "#" : continue
            #decode parameter line:
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

        #Add eval criterion:
        if __hparams["val_criterion"] is None and __hparams["criterion"] is not None :
            __hparams["val_criterion"] = __hparams["criterion"]
        #Save profile path to dict
        __hparams["profile_file"] = profile_path
        #Save info if there is a file with loadable model
        __hparams['load_model'] = True if load_model_path != "" else False
        __hparams['load_model_path'] = load_model_path
        #Check if all hparams are set:
        if any( elem is None for elem in __hparams.values()):
            print("Not all parameters set!")
            sys.exit(err.PROFILE_NOT_ALL_PARAMS_SET)
        return __hparams


