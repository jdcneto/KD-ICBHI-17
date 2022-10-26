# Summary
This repo contain the experiments in the project Lung Sound Classification using MobileVit and Knowledge Distillation.
1. The teacher model you can find [here](https://github.com/qiuqiangkong/audioset_tagging_cnn)
2. The weights you can download in [here](https://www.dropbox.com/sh/si61g3j69rvuw0w/AAAdHObq3_9G4QBKzgaJLIcUa?dl=0)   
- Create a folder called trained and download the weights on it.
3. Additional info about the ICBHI 2017 dataset You can be found [here](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)

# How to run
We disponibilize two notebooks:
1. In the Creating_Database.ipynb is show how preprocessing the dataset. You must run it before train the models.
2. The train.ipynb is a example of how to run the knoledge distillation. 
* Feel free to make changes and improvements.

# You may need to install:
1. torchlibrosa 0.0.9
2. pytorch 1.12.1+cu116
3. transformers 4.23.1

Have Fun!!!
