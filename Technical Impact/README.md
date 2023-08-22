# Technical Impact decision value
This folder contains documents related to the automating the technical impact decision point for SSVC scoring. It contains machine learning models for classifying the technical impacts of given vulnerabilities based on their vulnerability descriptions. 

## What are the classifiers used for?

### Binary Classifier
The binary classifier is a multi-modal binary classifier that utilizes text classification to take in Text (CVE descriptions typically) and output a Decision value based on the modes defined in the **MODES** dictionary at the top of the script. These modes are completely dynamic and should not have any problems creating and removing modes of the models. The only thing that is important to note is that the values are case sensitive to the data you feed it, so make sure that if you are putting in 'Partial' in the datafile, make sure to put 'Partial' in the modes dictionary.


In terms of data processing and input the code and model accepts intake of CSV files that are in the format of the following:

```
Text,Decision
Desc,total/decision1
Desc2,partial/decision0
```
It is important to note that these CSV headers are **case-sensitive**, and you will run into errors when training if you do not follow the casings above.

### Multiclass Classifier
The multiclass classifier is a multi-modal multiclass classifier that utilizes text classification to take in Text (CVE descriptions typically) and output a class based on the mode selected and mode defined in the **MODES** dictionary at the top of the script. These modes are completely dynamic and should not have any problems creating and removing modes of the models. The only thing that is important to note is that the values are case sensitive to the data you feed it, so make sure that if you are putting in 'HIGH' in the datafile, make sure to put 'HIGH' in the modes dictionary.


In terms of data processing and input the code and model accepts intake of CSV files that are in the format of the following:

(Example based on technical impact, but it would work the same with new modes you define)
```
Text,Class
Desc,LOW
Desc2,NONE
Desc3,HIGH
```
It is important to note that these CSV headers are **case-sensitive**, and you will run into errors when training if you do not follow the casings above.
Another requirement is that the classes should all be capitalized to run effectively with no issues. 

### Modes Dictionary Example
#### Binary
```python
# Binary modes do not have any more values other than 0 and 1
MODES = {
    "TECHNICAL_IMPACT": {
        0: "partial",
        1: "total"
    },
    "AUTOMATABILITY": {
        0: "NO",
        1: "YES"
    }
}
```

#### Multiclass
```python
# multiclass can have 0..n values
MODES = {
    "TECHNICAL_IMPACT": {
        0: "NONE",
        1: "LOW",
        2: "HIGH"
    },
    "AUTOMATABILITY": {
        0: "None",
        1: "Reconnaissance",
        2: "Weaponization",
        3: "Delivery"
    }
}
```

## Set up your environment to run

Firstly you need to download and install [Anaconda](https://www.anaconda.com/download)

Once installed open **Anaconda Navigator**, navigate to the environments tab on the left side of the navigator. 

Once there press the import button and import the yml that is found in the `environment` folder.
This will create the environment that was created and ran while developing these classifiers, and allow for GPU usage with Tensorflow if your computer supports it.

![anaconda import.png](environment%2Freadme_media%2Fanaconda%20import.png)


Once anaconda creates your environment, open your classifiers' repo directory in PyCharm and you will likely have a lot of errors, which is to be expected. You have to set your environment to the anaconda environment we just created.

In the bottom right of the IDE there will be an area where you may or may not have a Python interpreter setup. It will likely default to a version of Python you have installed.

Click Python 3.X or whichever version displayed (if there is one), once you do that a dropdown will appear like the following:

![add interpreter.png](environment%2Freadme_media%2Fadd%20interpreter.png)

Once it pops up, click Add New Interpreter, and then add local interpreter. A menu will pop up that will have options on the left, you are looking to click 
**Conda Environment** on the left side. It will then load anaconda environments automatically, and you should be able to find TF_GPU or whichever name that you chose for your 
environment. Click that and click apply, and it will then have your PyCharm index a lot of libraries, which will take a couple of minutes and then your code should be errorless and ready to run!

## Usage
In terms of using the code you have a relatively straight forward command line usage that is written out with examples of how to run things along with descriptions of each command.

There is one very important factor that needs to be addressed before your start having your computer start training models and running experiments.
The code is not supportive of file names with spaces or spaces in arguments.

We will take the Train and Test command for example.

The following example of a command input that is unsupported and will not work with the code currently in place. 

`enter a command: tat New Model training/new model data.csv 100 16`

The reason the command will not work because there are spaces with the model name along with the file directory. 

The following command is fully supported and is how you need to run commands (no matter the command) with the current version of the script
`enter a command: tat New_Model training/new_model_data.csv 100 16`

This command will train a model for 100 epochs with a 16 batch size, along with 5-fold cross validation. Good luck!
