## Trajectory is the Description of Arrival
> The final project of **SJTU AI3602** *Data Mining* course Group 23 
#### Introduction
Our project is focus on the prediction of taxi arrival by its trajectory (prefix) , the trip start time and some meta information like TAXI ID, CALL PHONE NUMBER and STAND ID, which is based on the competition held by ECML/PKDD on *kaggle* in 2015. The competition has ended so we just compare our result with the scores of the teams joining it. The dataset is available at [https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i).

#### Paper

Our paper for this project is available [here](https://github.com/COMoER/Trajectory-is-the-Description-of-Arrival/report.pdf), you can get more information of our work in this.

#### Dependency

>conda create -n trajectory python=3.8
>
>conda activate trajectory
>
>pip install -r requirements.txt

#### Usage of our code

Before all, you should download the dataset from *kaggle* [ECML/PKDD 15: Taxi Trajectory Prediction (I) | Kaggle](https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i/data) and save it as a directory named *dataset* in main context, that is, *dataset* has the same parent with *project*

Our main codes are in the directory *project*.
- *pretained* directory contains the pretrained binary file to successfully do preprecessing of the raw data.
- *utils* directory contains the supporting class for preprocessing and evaluation.
- *script* directory contains the train and generation(the submission of *kaggle*) script
- *model* directory contains the deep learning model

So just run the *train.py* script

> python project/script/train.py

The available options are

> --size N	the size of sampling from whole dataset
>
> --epoch E 	the max epoch
>
> --lr LR	the learning rate of Adam optimizer
>
> --random_length using random length training, default using partial mode
>
> --head change partial mode to head mode
>
> --prefix add prefix 5 point (x,y) position to input
>
> --meta add meta information (start time and other meta information) to input

The result will automatically be saved to the *log* directory with the training starting time as the directory name, which contains 

> args.yaml	contains the parameters selected above 
>
> log_train.txt	contains the training log like validation loss and arrival error
>
> model_best.pt the parameter of the model which has the best loss performance on validation
>
> model_last.pt the parameter of the model of the last epoch

To generate the submission of *kaggle*, you should first put the trained parameter into the *pretrained* directory and name it *model_best.pt*, then run

> python project/script/gen.py

add `--prefix` and `--meta` if you use them to train. 