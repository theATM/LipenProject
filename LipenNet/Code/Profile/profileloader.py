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

    #Training Parameters
    initial_learning_rate: float | None
    scheduler_list: list | None

    #Augmentation Parameters
    augmentation_type : en.AugmentationType | None




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
    elif set_type == list:
        return value[1:-1].split(",")
    elif set_type == en.DatasetName:
        return en.DatasetName[value]
    elif set_type == en.AugmentationType:
        return en.AugmentationType[value]

    else:
        print("Unimplemented Type Detected!")
        exit(err.PROFILE_UNIMPLEMENTED_TYPE)




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

    #Data Params
    "augmentation_type" : None,

    "initial_learning_rate" : None,
    "scheduler_list" : None,

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
                exit(err.PROFILE_EMPTY_PARAM_VALUE)
            parameter_key : str = line.split("=")[0]
            parameter_value = line.split("=")[1]
            if parameter_key in __hparams:
                parameter_value = convertStrToType(parameter_key,parameter_value)
                __hparams[parameter_key] = parameter_value
            else:
                print("Wrong key in profile.txt")
                exit(err.PROFILE_WRONG_KEY_IN_PROFILE_FILE)
        #Check if all hparams are set:
        if not all(__hparams.values()):
            print("Not all parameters set!")
            exit(err.PROFILE_NOT_ALL_PARAMS_SET)
        return __hparams


