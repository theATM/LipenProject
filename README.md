# LipenProject - School Equipment Classification Project
Hi this is a simple Machine Learning project design to test our skill in designing simple multiclas clasification project.

## Dataset
Firstly we created our own classification dataset.
Avaiable on the kaggle website under this link: https://kaggle.com/datasets/1a8d7ee76e6970626640cd7be11499d46409fe212e0397a2776d7e01cb84eee5
It contains images of 5 common school items:  pen, pencil, rubber, ruler, a set square or images containg neither.

![lipenclasses](https://github.com/theATM/LipenProject/assets/48883111/532d59fa-5696-4fd2-b047-9d1008e6d59e)


## Data Labeling
The LipenTagger script have been created. 
A tagger UI python app desing to speed up the labeling process.

The UI of the LipenTagger tool used to label images in dataset can be seen bellow (UI in Polish):
![image](https://github.com/theATM/LipenProject/assets/48883111/119d5978-ffa3-4cdc-aba6-ed37b20635ea)


## Neural Network
The LipenNet dir contains the training code used to conduct classification experiment.
Custom dataloader is proposed that utilises our custom csv dataformat.
Training and evaluation scripts are avaiable in the Functional subdir.
Two architectures were tested. A ResNet 18 and ConvNeXt both pretrained on the Imagenet dataset.

Different optimisation techinques such as early stopping, layer freezing and hard example mining were used.

### Result Table

| Testing Results | ResNet 18    | ConvNeXt    |
| :---:   | :---: | :---: |
| Top1 Accuracy | 61.54%   | 77.31%   |
| Top2 Accuracy | 83.08%   | 92.31%   |
| Top3 Accuracy | 89.23%   | 98.08%   |
| F1 Score      | 0.61     | 0.77     |

### Confusion Matrix (on Test data)

| ResNet 18    | ConvNeXt    |
| :---: | :---: |
| ![image](https://github.com/theATM/LipenProject/assets/48883111/6979224f-27f3-4619-a08b-3a042a09f090) | ![image](https://github.com/theATM/LipenProject/assets/48883111/7caf01df-4569-49a0-b056-5e4c722136c6) |

## Further Work

This project has been used as a basis for the model optimisation study.
In  <a href="https://github.com/Nikodemmn1/LipenOptimization"> (THIS)</a> project it has been shown that this model can be 80% pruned (reduced) without a significant drop in the model accuracy.



## References:

https://pytorch.org/vision/main/models/generated/torchvision.models.convnext_small.html#torchvision.models.convnext_small

https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html

## Technology
Python 3.10.4
