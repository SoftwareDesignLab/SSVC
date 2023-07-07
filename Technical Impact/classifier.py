import os

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import re
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
import statistics
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.model_selection import KFold

PREPROCESS_SEQ_LENGTH = 128

def trim_data(file, export_changes=False):
    """
    A preprocessing function that intakes a pandas dataframe of training data
    and then does stopword removal to help accuracy and returns the dataframe
    :param file: data file being used
    :param export_changes: whether the changes should be exported and show a side-by-side of changes
    :return:
    """
    data = pd.read_csv(file)
    # If requesting changes exported, there will be a new column created
    # that is a copy of the unchanged data.
    if export_changes:
        data['old_text'] = data.loc[:, 'Text']

    # download stopwords
    nltk.download('stopwords')

    sw = set(stopwords.words('english'))

    for i, row in data.iterrows():
        text = row['Text']
        words = text.split()
        filtered = ""

        for w in words:
            if w.lower() not in sw:
                filtered += w + " "

        filtered = filtered.strip()
        data.at[i, 'Text'] = filtered

    if export_changes:
        # make dir for files
        try:
            os.mkdir("./preprocessed")
        except:
            pass
        # export
        data.to_csv(f"./preprocessed/{file.replace('.csv', '_pre.csv')}", index=False)

    # Return the dataframe for the training to use
    return data

def get_bert_model():
    """
    Creates and compiles a BERT Classification model that would be used for text classification
    :return:
    """
    # Download premade preprocessors for BERT and apply them
    bert_preprocess = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
    bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")
    text_input = [tf.keras.layers.Input(shape=(), dtype=tf.string)]
    tokenize = hub.KerasLayer(bert_preprocess.tokenize)
    tokenized_inputs = [tokenize(segment) for segment in text_input]

    bert_pack_inputs = hub.KerasLayer(
        bert_preprocess.bert_pack_inputs,
        arguments=dict(seq_length=PREPROCESS_SEQ_LENGTH))
    encoder_inputs = bert_pack_inputs(tokenized_inputs)
    outputs = bert_encoder(encoder_inputs)

    # Init neural network layers
    dropout = 0.1
    l = tf.keras.layers.Dropout(dropout, name="dropout")(outputs['pooled_output'])
    l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

    model = tf.keras.Model(inputs=text_input, outputs=[l])
    # model metrics
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve='ROC',
            summation_method='interpolation',
            name=None,
            dtype=None,
            thresholds=None,
            multi_label=False,
            num_labels=None,
            label_weights=None,
            from_logits=False
        )
    ]

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=metrics)

    return model


def train_model(file_name, training_name, epochs, batch_size=32, downsample=False, stopword=True):
    """
    Function that trains a new model using a bert classifier from intake of a csv file
    with the format of CVE Descriptions (C I Description) and Tech Impact (total or partial)
    and would then train a model with user supplied amount of epochs
    :param file_name: csv file
    :param training_name: name of the model
    :param epochs: amount of epochs run
    :param batch_size: batch size of the training
    :param downsample: whether the trainer should sample the count of classifications
    :return:
    """

    # Send the data through stopword removal
    if stopword:
        df = trim_data(file_name)
    else:
        df = pd.read_csv(file_name)

    # Split into dataframes for partial or total impact
    df_partial = df[df['Impact'] == 'partial']
    df_total = df[df['Impact'] == 'total']

    # Downsampling total to match partial amounts
    if downsample:
        df_total = df_total.sample(df_partial.shape[0])

    # Create unified balanced dataframe
    df_balanced = pd.concat([df_total, df_partial])
    df_balanced['total'] = df_balanced['Impact'].apply(lambda x: 1 if x == 'total' else 0)

    # x = C I Desc
    # y = partial/total
    #x_train, x_test, y_train, y_test = train_test_split(df_balanced['Text'], df_balanced['total'],
     #                                                   stratify=df_balanced['total'], train_size=0.8)

    # vk
    # df_balanced['Text'] is each description on a separate line (100 lines)
    # df_balanced['total'] is each label on a separate line (100 lines)
    #print(df_balanced['total'])

    # define inputs and targets
    # x_train, x_test, y_train, y_test = train_test_split will not be used in kfold
    inputs = df_balanced['Text']
    targets = df_balanced['total']

    # define k-fold cross validation
    kf = KFold(n_splits=5,shuffle=True)

    # doing k-fold cross validation
    fold_no = 1
    for train, test in kf.split(inputs, targets):
        print("running fold #", fold_no)

        model = get_bert_model()

        # fit data
        print("Fitting the model with training data")
        # todo: add logger
        # use input[train] and targets[train] instead of x_train, x_test, y_train, y_test
        model.fit(inputs[train], targets[train], epochs=epochs, batch_size=batch_size)
        #print(inputs[train]) # length of this is 80... as it should be!
        #print(targets[train])  # length of this is 80... as it should be!

        print("Evaluating model")
        # use input[test] and targets[test] instead of x_train, x_test, y_train, y_test
        evaluated = model.evaluate(inputs[test], targets[test], batch_size=batch_size)
        #print(inputs[test]) # length of this is 20... as it should be!
        #print(targets[test])  # length of this is 20... as it should be!

        # todo: get metrics to save for each fold? or across all folds?

        fold_no += 1
    # stuff after this is train_test_split() stuff (not kfold)
    # may fail after this due to change in

    # it was here during train_test_split
    #model = get_bert_model()


    # make working dir
    try:
        os.mkdir(f"./models/{training_name}")
    except FileExistsError:
        print("A model with that name already exists.")
        return None

    # Create logger callback
    logger = CSVLogger(f"./models/{training_name}/log.csv", append=True, separator=',')

    # Fit model
    print("Fitting the model with training data")
    model.fit(x_train, y_train, epochs=epochs, callbacks=[logger], batch_size=batch_size)

    y_predicted = model.predict(x_test)
    y_predicted = y_predicted.flatten()
    # Set prediction to total if > confidence_interval, else partial (total = 1, partial = 0)
    confidence_interval = 0.55
    y_predicted = np.where(y_predicted > confidence_interval, 1, 0)

    # Save the model
    model.save(f"./models/{training_name}")
    # Create data obj for pandas dataframe
    data = []
    for i, cve in enumerate(x_test):
        # create string of total or partial based on prediction value (1 = total, 0 = partial)
        impact = "total" if y_predicted[i] == 1 else "partial"
        data.append([cve, impact])

    #Create predictions data frame
    pdf = pd.DataFrame(data, columns=["description", "impact"])
    pdf.to_csv(f"./models/{training_name}/predictions.csv", index=False)
    print("Created CSV File for the predictions made in this training.")

    # calc f1 score
    calculate_f1_score(f"./models/{training_name}/log.csv")
    # test predictions evaluation
    evaluated = model.evaluate(x_test, y_test, batch_size=batch_size)

    # assign statistics of prediction
    loss = evaluated[0]
    accuracy = evaluated[1]
    precision = evaluated[2]
    recall = evaluated[3]
    auc = evaluated[4]
    f1_score = 2 * (precision * recall) / (precision + recall)

    predictions_data = {
        "loss": loss,
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    predictions_log = pd.DataFrame(columns=["loss", "accuracy", "auc", "precision", "recall", "f1_score"],
                                   data=predictions_data, index=[0])

    predictions_log.to_csv(f"models/{training_name}/predictions_log.csv", index=False)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] The training of {training_name} has completed")

def print_help():
    """
    Function that prints out the commands for the script
    :return:
    """
    print("SSVC Technical Impact classifier (BERT):")
    print("\ttrain <model_name> <file> <epochs_run> (batch_size: default 32)- trains a new model from csv file with user specified epochs")
    print("\t\tex: train 100_examples 100_examples_even.csv 100")
    # print("")
    # print("\tevaluate <file> <model_name> - evaluates a file with a certain model")
    # print("\t\tex: evaluate 389_cves most_accurate_model")
    print("")
    print("\tanalyze <file_name> - reads a csv file and produces information relating to word counts")
    print("\t\tex: analyze 100_samples.csv")
    print("")
    print("\tquit - terminates program")

def calculate_f1_score(file):
    """
    Loads a CSV file of training statistics and calculates and appends F1 score to CSV
    :param file:
    :return:
    """
    # read the csv info pandas
    df = pd.read_csv(file, index_col=[0])
    df.reset_index()
    # create a list of the f1 scores that will be written into the csv
    f1_scores = []
    # calculate f1_score for each row and append to list
    for index, row in df.iterrows():
        precision = row["precision"]
        recall = row["recall"]
        f1_score = 2*(precision*recall)/(precision+recall)
        f1_scores.append(f1_score)

    # create new column
    df['f1_score'] = f1_scores

    # export new file
    df.to_csv(file)

def analyze_file(file):
    """
    takes in a csv file and finds the largest word count
    :param file:
    :return:
    """
    df = pd.read_csv(file)
    largest = 0
    over_128 = 0
    total = 0
    count = 0

    # list of word counts
    word_counts = []

    # list of cve desc >= 128
    long_cves = []
    for i,row in df.iterrows():
        # assign variables
        text = row["Text"]
        words = text.split(' ')
        word_count = len(words)
        # assign new largest if applicable
        if word_count > largest:
            largest = word_count

        # add for average stat
        total += word_count
        count += 1

        # If word >= 128 (pre processor max seq length) add to count
        if word_count >= PREPROCESS_SEQ_LENGTH:
            over_128 += 1
            long_cves.append(text)
        word_counts.append(word_count)

    average = total / count
    text_info = pd.DataFrame(columns=["over_128", "largest_count", "average_count", "median"], index=None)
    text_info = text_info.append({"over_128": over_128, "largest_count": largest, "average_count": average, "median": statistics.median(word_counts)}, ignore_index=True)
    long_df = pd.DataFrame(data=long_cves)
    try:
        # make dir if doesnt exist
        os.mkdir("./data_analysis")
    except FileExistsError:
        pass

    # get just the filename
    file = re.sub(".*/", "", file)

    text_info.to_csv(f'./data_analysis/{file.replace(".csv", "_word_data.csv")}', index=False)
    long_df.to_csv(f'./data_analysis/{file.replace(".csv", "_long_cve.csv")}', index=False)

def analyze_predictions(training_file_path, predictions_file_path, create_new_file=True, trim_descs=True):
    """
    Loads the training file and the predictions file of a model and determines the predictions
    correctness via another row given.
    :param training_file_path: trainings files
    :param predictions_file_path: predictions file
    :param create_new_file: boolean representing whether a separate file should be produced to show prediction correctness
    :return:
    """
    # read the csvs into pandas
    if trim_descs:
        training_file = trim_data(training_file_path)  # have to send through pre processo
    else:
        training_file = pd.read_csv(training_file_path)
    training_file.reset_index()

    predictions_file = pd.read_csv(predictions_file_path)
    predictions_file.reset_index()
    # create a list of the correct or incorrect
    correct = []
    # calculate f1_score for each row and append to list
    for index, row in predictions_file.iterrows():
        desc = row["description"]
        impact = row["impact"]

        search = training_file[(training_file["Text"] == desc)]
        train_data_impact = search.iloc[0]['Impact']

        # add result to list based on if search and impact match
        correct.append("correct" if str(train_data_impact).lower() == impact.lower() else "incorrect")
    # create new column
    predictions_file['correct'] = correct

    # export new file
    if create_new_file:
        predictions_file.to_csv(predictions_file_path.replace(".csv", "_checked.csv"), index=False)
    else:
        # Write to already created predictions file
        predictions_file.to_csv(predictions_file_path, index=False)

def create_final_score(checked_predictions_file):
    """
    Creates and exports the final score based off a checked predictions file
    Final score icludes accuracy, precision, recall, f1 score
    :param checked_predictions_file:
    :return:
    """
    predictions = pd.read_csv(checked_predictions_file)

    tp = 0  # true positive - correct predicted as total
    tn = 0  # true negative - correct predicted as partial
    fp = 0  # false positive - predicted as total should be partial
    fn = 0  # false negative - partial should be total

    for index, row in predictions.iterrows():
        decision = row['impact']
        correct = row['correct']

        if decision == "total" and correct == "correct":
            # True positive
            tp += 1
        elif decision == "partial" and correct == "correct":
            # True negative
            tn += 1
        elif decision == "total" and correct == "incorrect":
            # false positive
            fp += 1
        elif decision == "partial" and correct == "incorrect":
            # false negative
            fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # data to put into csv file
    data = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    log = pd.DataFrame(columns=["accuracy", "precision", "recall", "f1_score"], index=None)
    log = log.append(data, ignore_index=True)
    log.to_csv(checked_predictions_file.replace(".csv", "_log.csv"))

def check_positives(checked_predictions_file):
    """
    Creates and exports a file that contains true positives, true negatives etc.
    based off a checked predictions file
    Final score icludes accuracy, precision, recall, f1 score
    :param checked_predictions_file: predictions file
    :return:
    """
    predictions = pd.read_csv(checked_predictions_file)

    tp = 0  # true positive - correct predicted as total
    tn = 0  # true negative - correct predicted as partial
    fp = 0  # false positive - predicted as total should be partial
    fn = 0  # false negative - partial should be total

    for index, row in predictions.iterrows():
        decision = row['impact']
        correct = row['correct']

        if decision == "total" and correct == "correct":
            # True positive
            tp += 1
        elif decision == "partial" and correct == "correct":
            # True negative
            tn += 1
        elif decision == "total" and correct == "incorrect":
            # false positive
            fp += 1
        elif decision == "partial" and correct == "incorrect":
            # false negative
            fn += 1

    # data to put into csv file
    tp_data = {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

    tp_log = pd.DataFrame(columns=["tp", "tn", "fp", "fn"], index=None)
    tp_log = tp_log.append(tp_data, ignore_index=True)
    tp_log.to_csv(checked_predictions_file.replace(".csv", "_positives_log.csv"))


def main():
    """
    Main function that runs the program
    :return:
    """
    while True:
        # print help
        print_help()

        # intake command from user and split it into its arguments
        command = str(input("enter a commend: "))
        args = command.strip().split(" ")

        # Switch statement of first argument of command to process the commnad in the correct way
        match args[0]:
            case "train":
                model_name = None
                file_name = None
                epochs = None
                batch_size = None
                stop_word = None
                try:
                    model_name = args[1]
                    file_name = args[2]
                    epochs = int(args[3])
                    batch_size = None
                    # Do a separate try except for Batch Size so that it can be skipped
                    try:
                        batch_size = int(args[4])
                    except Exception:
                        # if an exception occurs, regardless the type, set to default batch of 32
                        batch_size = 32

                    try:
                        stop = args[5]
                        if stop.lower() == "y":
                            stop_word = True
                        else:
                            stop_word = False
                    except Exception:
                        # default to no stopword removal
                        stop_word = False

                except FileNotFoundError as e:
                    print("Training file not found.")
                except ValueError as e:
                    print("You must enter a valid number for the amount of epochs youd like to run")
                except IndexError:
                    print("You must enter all necessary arguments")

                if file_name is None or model_name is None or epochs is None:
                    # Error message if args are missing
                    print("You must enter all arguments for commands")
                else:
                    # Train the model
                    print(f"Training new model named {model_name} for {epochs} epochs, with a {batch_size} batch size.")
                    print(f"Stopword removal is {'enabled' if stop_word else 'disabled'}.")
                    train_model(file_name, model_name, epochs, batch_size, stopword=stop_word)
                    print("Adding correctness of predictions")
                    analyze_predictions(file_name, f'models/{model_name}/predictions.csv', create_new_file=False, trim_descs=stop_word)
                    check_positives(f'models/{model_name}/predictions.csv')
            case "analyze":
                file_name = args[1]
                analyze_file(file_name)
                print("Analysis has completed")
            case "check":
                training_file = args[1]
                predictions_file = args[2]
                trim = args[2]

                trim_desc = None
                if trim.lower() == 'y':
                    trim_desc = True
                else:
                    trim_desc = False

                analyze_predictions(training_file, predictions_file, trim_descs=trim_desc, create_new_file=False)
                print(f"Checked and exported the file to {predictions_file.replace('.csv', '_checked.csv')}")
            case "fscore":
                predictions_file = args[1]
                create_final_score(predictions_file)
                print(f"Created final score log at {predictions_file.replace('.csv', '_log.csv')}")
            case "preproc":
                file = args[1]
                try:
                    trim_data(file, True)
                except FileNotFoundError:
                    print("You must enter a valid file name!")
            case "positives":
                file = args[1]
                try:
                    check_positives(file)
                    print("Exported checked file containing true positives,true negatives, false positives, false negatives")
                except FileNotFoundError:
                    print("You must enter a valid file name!")
            case "quit":
                break


if __name__ == "__main__":
    main()