import os
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import re
from keras.callbacks import CSVLogger
from datetime import datetime
import numpy as np
import statistics
import nltk
from keras.optimizers import Adam
from keras.utils import to_categorical
from nltk.corpus import stopwords
# noinspection PyUnresolvedReferences
import tensorflow_text as text  # do not remove this import, it is required for hub.load()
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
        except FileExistsError:
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
    # Download pre-made preprocessors for BERT and apply them
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
    layer = tf.keras.layers.Dropout(dropout, name="dropout")(outputs['pooled_output'])
    # changed sigmoid to softmax for multi class
    layer = tf.keras.layers.Dense(3, activation='softmax', name="output")(layer)

    model = tf.keras.Model(inputs=text_input, outputs=[layer])
    # model metrics
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
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

    # changed loss from binary crossentropy to categorical
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=metrics)

    return model


def train_model(file_name, training_name, epochs, batch_size=32, downsample=True, stopword=False, folds=5):
    """
    Function that trains a new model using a bert classifier from intake of a csv file
    with the format of CVE Descriptions (C I Description) and Tech Class (total or partial)
    and would then train a model with user supplied amount of epochs
    :param file_name: csv file
    :param training_name: name of the model
    :param epochs: amount of epochs run
    :param batch_size: batch size of the training
    :param downsample: whether the trainer should sample the count of classifications
    :param stopword: whether stopword removal should be used
    :param folds: how many folds for k fold cross validation
    :return:
    """

    # Send the data through stopword removal
    if stopword:
        df = trim_data(file_name)
    else:
        df = pd.read_csv(file_name)

    # Split into dataframes for none low high df
    df_none = df[df['Class'] == 'NONE']
    df_low = df[df['Class'] == 'LOW']
    df_high = df[df['Class'] == 'HIGH']

    # assigning size of the samples in a dict to determine
    # which is lowest (for downsampling purposes
    raw_class_sizes = {'NONE': len(df_none.index),
                       'LOW': len(df_low.index),
                       'HIGH': len(df_high.index)}

    min_num_classes = min(raw_class_sizes.values())

    # Downsampling the dataset to an equal amount of samples.
    if downsample:
        print(f"Downsampling is enabled! The samples will be cut to {min_num_classes} samples of each class.")
        for key in ["NONE", "LOW", "HIGH"]:
            if min_num_classes == raw_class_sizes[key]:
                print(f"The sample number is based on the lowest class size available. In this sample the lowest class size available is {key}")
                if key == "NONE":
                    df_low = df_low.sample(n=raw_class_sizes['NONE'])
                    df_high = df_high.sample(n=raw_class_sizes['NONE'])
                elif key == "LOW":
                    df_none = df_none.sample(n=raw_class_sizes['LOW'])
                    df_high = df_high.sample(n=raw_class_sizes['LOW'])
                elif key == "HIGH":
                    df_low = df_low.sample(n=raw_class_sizes['HIGH'])
                    df_none = df_none.sample(n=raw_class_sizes['HIGH'])

    # Create unified balanced dataframe
    df_balanced = pd.concat([df_none, df_low, df_high])

    # reindex the concatanated dataframe so the inputs work properly.
    df_balanced = df_balanced.reset_index(drop=True)
    df_balanced["total"] = df_balanced['Class'].map({"NONE": 0,
                                                     "LOW": 1,
                                                     "HIGH": 2})
    inputs = df_balanced['Text']
    targets = to_categorical(df_balanced['total'], 3)
    # make working dir
    try:
        os.mkdir(f"./models/{training_name}")
    except FileExistsError:
        print("A model with that name already exists.")
        return None

    # define k-fold cross validation
    kf = KFold(n_splits=folds, shuffle=True)

    # doing k-fold cross validation
    fold_no = 1
    fold_scores = []
    model = get_bert_model()

    for train, test in kf.split(inputs, targets):
        in_train = inputs[train]
        targets_train = targets[train]

        in_test = inputs[test]
        targets_test = targets[test]

        # make fold working dir
        try:
            os.mkdir(f"./models/{training_name}/fold_{fold_no}")
        except FileExistsError:
            print("A fold folder for this fold already exists.")
            return None
        print(f"Running Fold {fold_no}/{folds}")

        # fit data
        print(f"(FOLD {fold_no}) Fitting the model with training data")
        logger = CSVLogger(f"./models/{training_name}/fold_{fold_no}/log.csv", append=True, separator=',')

        # use input[train] and targets[train] instead of x_train, x_test, y_train, y_test
        model.fit(in_train, targets_train, epochs=epochs, callbacks=[logger], batch_size=batch_size)

        print(f"(FOLD {fold_no}) Evaluating model")
        y_predicted = model.predict(in_test)

        max_values = y_predicted.max(axis=1).reshape(-1, 1)
        y_predicted = np.where(y_predicted == max_values, 1, 0)
        # print(y_predicted)

        # Save the model for this fold
        model.save(f"./models/{training_name}/fold_{fold_no}")
        # Create data obj for pandas dataframe
        data = []
        for i, cve in enumerate(in_test):
            pred = y_predicted[i]
            none_array = [1, 0, 0]
            low_array = [0, 1, 0]
            high_array = [0, 0, 1]
            if np.array_equal(pred, none_array):
                data.append([cve, "NONE"])
            elif np.array_equal(pred, low_array):
                data.append([cve, "LOW"])
            elif np.array_equal(pred, high_array):
                data.append([cve, "HIGH"])
            else:
                # assign improper prediction if there is predictions that classified into a non-accepted format
                data.append([cve, f"IMPROPER PREDICTION {pred}"])

        # Create predictions data frame
        pdf = pd.DataFrame(data, columns=["Description", "Class"])
        pdf.to_csv(f"./models/{training_name}/fold_{fold_no}/predictions.csv", index=False)
        print(f"(FOLD {fold_no}) CSV File for the predictions made in this training have been exported.")
        predictions_data = analyze_predictions(file_name, f'models/{training_name}/fold_{fold_no}/predictions.csv',
                                               create_new_file=False,
                                               trim_descs=stopword)

        # calc f1 score
        # test predictions evaluation
        evaluated = model.evaluate(in_test, targets_test, batch_size=batch_size)
        evaluated.extend(predictions_data.values())
        fold_scores.append(evaluated)

        # assign statistics of prediction
        predictions_data = list_to_data_dict(evaluated)

        predictions_log = pd.DataFrame(columns=["loss", "accuracy", "auc",
                                                "none_precision", "none_recall", "none_f1",
                                                "low_precision", "low_recall", "low_f1",
                                                "high_precision", "high_recall", "high_f1",
                                                "macro_precision", "macro_recall", "macro_f1"],
                                       data=predictions_data, index=[0])

        predictions_log.to_csv(f"models/{training_name}/fold_{fold_no}/predictions_log.csv", index=False)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] (FOLD {fold_no}) The training and testing of {training_name} has completed")
        fold_no += 1

    print(f"[{datetime.now().strftime('%H:%M:%S')}] All folds for {training_name} have completed")

    # average all stats, put them into a dictionary and export from a dataframe
    avg_scores = average_fold_stats(fold_scores)
    avg_data = list_to_data_dict(avg_scores)
    predictions_log = pd.DataFrame(columns=["loss", "accuracy", "auc",
                                            "none_precision", "none_recall", "none_f1",
                                            "low_precision", "low_recall", "low_f1",
                                            "high_precision", "high_recall", "high_f1",
                                            "macro_precision", "macro_recall", "macro_f1"],
                                   data=avg_data, index=[0])
    predictions_log.to_csv(f"models/{training_name}/{folds}_fold_average_log.csv", index=False)


def list_to_data_dict(stats_list):
    """
    Takes in a list of stats then returns a dictionary with headers
    for the relevant stats from testing of models
    :param stats_list: list of stats
    :return: dictionary of scores
    """
    loss = stats_list[0]
    accuracy = stats_list[1]
    auc = stats_list[2]
    none_precision = stats_list[3]
    none_recall = stats_list[4]
    none_f1 = stats_list[5]
    low_precision = stats_list[6]
    low_recall = stats_list[7]
    low_f1 = stats_list[8]
    high_precision = stats_list[9]
    high_recall = stats_list[10]
    high_f1 = stats_list[11]
    macro_precision_average = stats_list[12]
    macro_recall_average = stats_list[13]
    macro_f1_average = stats_list[14]

    data = {
        "loss": loss,
        "accuracy": accuracy,
        "auc": auc,
        "none_precision": none_precision,
        "none_recall": none_recall,
        "none_f1": none_f1,
        "low_precision": low_precision,
        "low_recall": low_recall,
        "low_f1": low_f1,
        "high_precision": high_precision,
        "high_recall": high_recall,
        "high_f1": high_f1,
        "macro_precision": macro_precision_average,
        "macro_recall": macro_recall_average,
        "macro_f1": macro_f1_average
    }

    return data


def print_help():
    """
    Function that prints out the commands for the script
    :return:
    """
    print("")
    print("SSVC Technical Impact Confidentiality/Integrity Class classifier (BERT):")
    print_command("Train and Test",
                  "Trains and tests a new model from a csv file using user specified input and 5 fold cross validation.",
                  "tat <model_name> <train_file> <epochs> (batch_size: default 32) (stopword: y/n)",
                  "tat 100_examples 100_examples_even.csv 100 16")
    print("")
    print_command("Analyze Training Wordcount",
                  "Reads a training file and displays information relating to wordcounts of training data",
                  "wordcount <file_name>", "wordcount 100_epochs_training.csv")
    print("")
    print_command("Check Predictions", "Checks an unchecked predictions file with its associated training file.",
                  "check <training_file> <unchecked_predictions_file> (stopword: y/n)",
                  "check 100training.csv model/100training.csv")
    print("")
    print_command("Display Preprocesses Training Data", "Preprocesses and exports a training file",
                  "preproc <training_file>", "preproc 1000_examples.csv")
    print("")
    print("Quit - terminates program")
    print("")


def print_command(name, description, syntax, example):
    """
    Easy util to print commands out easily and formatted
    :param name: name of the command i.e. Train and Test
    :param description: description of the command i.e. Trains and Tests a new model
    :param syntax:  syntax of the command i.e. tat <model_name> <file> <epochs> (batch_size: default 32)
    :param example: tat 100_ex 100_examples.csv 100 16
    :return:
    """
    print(name)
    print(f"\tDescription: {description}")
    print(f"\tSyntax: {syntax}")
    print(f"\t\tex: {example}")


def average_fold_stats(lists):
    """
    Function that takes in a list of lists of stats from kfold cross validation and averages them to create an average
    score for the k-fold cross validation
    :param lists: lists of stats
    :return: average list
    """
    num_lists = len(lists)
    list_length = len(lists[0])  # Assuming all sublists have the same length

    averaged_list = []
    for i in range(list_length):
        values = [sublist[i] for sublist in lists]  # Get the values at index i from all sublists
        average = sum(values) / num_lists  # Calculate the average
        averaged_list.append(average)

    return averaged_list


def analyze_file_wordcounts(file):
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
    for i, row in df.iterrows():
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

        # If word >= 128 (preprocessor max seq length) add to count
        if word_count >= PREPROCESS_SEQ_LENGTH:
            over_128 += 1
            long_cves.append(text)
        word_counts.append(word_count)

    average = total / count
    text_info = pd.DataFrame(columns=["over_128", "largest_count", "average_count", "median"], index=None)
    text_info = text_info.append({"over_128": over_128, "largest_count": largest, "average_count": average,
                                  "median": statistics.median(word_counts)}, ignore_index=True)
    long_df = pd.DataFrame(data=long_cves)
    safe_mk_dir("./data_analysis")

    # get just the filename
    file = re.sub(".*/", "", file)

    text_info.to_csv(f'./data_analysis/{file.replace(".csv", "_word_data.csv")}', index=False)
    long_df.to_csv(f'./data_analysis/{file.replace(".csv", "_long_cve.csv")}', index=False)


def safe_mk_dir(path):
    """
    Helping method to create directories and ignore if they already exist
    this method is mainly for setting up directories for things like models, preprocessed data etc.
    :param path: where to create the directory
    :return: None
    """
    try:
        # make dir if it doesn't exist
        os.mkdir(path)
    except FileExistsError:
        pass


def safe_divide(numerator, denominator):
    """
    A helper function that is used to safely divide numbers
    specifically in use of precision, recall, f1 stats or other relevant uses.
    :param numerator: numerator
    :param denominator: denomenator
    :return: result or 0 if divbyzero
    """
    try:
        result = numerator / denominator
    except ZeroDivisionError:
        result = 0
    return result


def analyze_predictions(training_file_path, predictions_file_path, create_new_file=True, trim_descs=True):
    """
    Loads the training file and the predictions file of a model and determines the predictions
    correctness via another row given.

    it also calculates the true positives, negatives, false positives, false negatives for each class
    and exports them alongside.
    
    Calculates the precision, recall and f1 scores along with macro average of each stat. Once the calculations 
    complete a dictionary is created and returned 
    
    :param training_file_path: trainings files
    :param predictions_file_path: predictions file
    :param create_new_file: boolean representing whether a separate file should be produced to show prediction correctness
    :param trim_descs: Whether the analysis trims descriptions or not
    :return eval_data: dictionary of all data created
    """
    # read the csvs into pandas
    if trim_descs:
        training_file = trim_data(training_file_path)  # have to send through preprocessor
    else:
        training_file = pd.read_csv(training_file_path)
    training_file.reset_index()

    predictions_file = pd.read_csv(predictions_file_path)
    predictions_file.reset_index()
    # create a list of the correct or incorrect
    correct = []
    # calculate f1_score for each row and append to list

    # initialize variables for positives
    # none positives
    none_tp = 0  # true positive - correct predicted as total
    none_tn = 0  # true negative - correct predicted as partial
    none_fp = 0  # false positive - predicted as total should be partial
    none_fn = 0  # false negative - partial should be total

    # low positives
    low_tp = 0  # true positive - correct predicted as total
    low_tn = 0  # true negative - correct predicted as partial
    low_fp = 0  # false positive - predicted as total should be partial
    low_fn = 0  # false negative - partial should be total

    # high positives
    high_tp = 0  # true positive - correct predicted as total
    high_tn = 0  # true negative - correct predicted as partial
    high_fp = 0  # false positive - predicted as total should be partial
    high_fn = 0  # false negative - partial should be total

    for index, row in predictions_file.iterrows():
        desc = row["Description"]
        guessed_ci_class = row["Class"]

        search = training_file[(training_file["Text"] == desc)]
        train_data_class = search.iloc[0]['Class']
        correct_class = str(train_data_class)

        # add result to list based on if search and Class match
        true_correct = correct_class.lower() == guessed_ci_class.lower()
        correct.append("correct" if true_correct else "incorrect")

        # None true positives
        if guessed_ci_class == "NONE" and true_correct:
            # is predicted as NONE and supposed to be NONE
            none_tp += 1
        elif not guessed_ci_class == "NONE" and not correct_class == "NONE":
            # is not predicted as none, and the correct answer is not none
            none_tn += 1
        elif guessed_ci_class == "NONE" and not correct_class == "NONE":
            # is predicted as none, correct answer is not none.
            none_fp += 1
        elif not guessed_ci_class == "NONE" and correct_class == "NONE":
            # is predicted as HIGH or LOW, supposed to be NONE
            none_fn += 1

        # Low true positives
        if guessed_ci_class == "LOW" and true_correct:
            # is predicted as LOW and supposed to be LOW
            low_tp += 1
        elif not guessed_ci_class == "LOW" and not correct_class == "LOW":
            # is not predicted as low, and the correct answer is not low
            low_tn += 1
        elif guessed_ci_class == "LOW" and not correct_class == "LOW":
            # is predicted as low, correct answer is not low.
            low_fp += 1
        elif not guessed_ci_class == "LOW" and correct_class == "LOW":
            # is predicted as HIGH or NONE, supposed to be LOW
            low_fn += 1

        # High true positives
        if guessed_ci_class == "HIGH" and true_correct:
            # is predicted as HIGH and supposed to be HIGH
            high_tp += 1
        elif not guessed_ci_class == "HIGH" and not correct_class == "HIGH":
            # is not predicted as HIGH, and the correct answer is not HIGH
            high_tn += 1
        elif guessed_ci_class == "HIGH" and not correct_class == "HIGH":
            # is predicted as HIGH, correct answer is not HIGH.
            high_fp += 1
        elif not guessed_ci_class == "HIGH" and correct_class == "HIGH":
            # is predicted as NONE or LOW, supposed to be HIGH
            high_fn += 1

    # create new column
    predictions_file['correct'] = correct

    positives_header = ["none_tp", "none_tn", "none_fp", "none_fn", "low_tp", "low_tn", "low_fp", "low_fn",
                        "high_tp", "high_tn", "high_fp", "high_fn"]
    positives_data = {
        "none_tp": none_tp,
        "none_tn": none_tn,
        "none_fp": none_fp,
        "none_fn": none_fn,
        "low_tp": low_tp,
        "low_tn": low_tn,
        "low_fp": low_fp,
        "low_fn": low_fn,
        "high_tp": high_tp,
        "high_tn": high_tn,
        "high_fp": high_fp,
        "high_fn": high_fn
    }

    # Create stats for each class
    none_precision = safe_divide(none_tp, none_tp + none_fp)
    none_recall = safe_divide(none_tp, none_tp + none_fn)
    none_f1 = safe_divide(2 * none_precision * none_recall, (none_precision + none_recall))

    low_precision = safe_divide(low_tp, low_tp + low_fp)
    low_recall = safe_divide(low_tp, (low_tp + low_fn))
    low_f1 = safe_divide(2 * low_precision * low_recall, low_precision + low_recall)

    high_precision = safe_divide(high_tp, (high_tp + high_fp))
    high_recall = safe_divide(high_tp, high_tp + high_fn)
    high_f1 = safe_divide(2 * high_precision * high_recall, high_precision + high_recall)

    macro_precision_average = (none_precision + low_precision + high_precision) / 3
    macro_recall_average = (none_recall + low_recall + high_recall) / 3
    macro_f1_average = (none_f1 + low_f1 + high_f1) / 3

    # assign to dict that will be returned 
    eval_data = {
        "none_precision": none_precision,
        "none_recall": none_recall,
        "none_f1": none_f1,
        "low_precision": low_precision,
        "low_recall": low_recall,
        "low_f1": low_f1,
        "high_precision": high_precision,
        "high_recall": high_recall,
        "high_f1": high_f1,
        "macro_precision": macro_precision_average,
        "macro_recall": macro_recall_average,
        "macro_f1": macro_f1_average
    }

    # export positives information
    positives_frame = pd.DataFrame(columns=positives_header, data=positives_data, index=[0])
    positives_frame.to_csv(predictions_file_path.replace(".csv", "_positives.csv"), index=False)

    # export new file
    if create_new_file:
        predictions_file.to_csv(predictions_file_path.replace(".csv", "_checked.csv"), index=False)
    else:
        # Write to already created predictions file
        predictions_file.to_csv(predictions_file_path, index=False)

    return eval_data


def main():
    """
    Main function that runs the program
    :return:
    """

    # housekeeping checks (makes sure all proper folders are made)
    safe_mk_dir(f"./models")
    safe_mk_dir(f"./preprocessed")
    safe_mk_dir(f"./data_analysis")

    while True:
        # print help
        print_help()

        # intake command from user and split it into its arguments
        command = str(input("enter a command: "))
        args = command.strip().split(" ")

        if args[0].lower() == "tat":
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
                except IndexError:
                    # if an exception occurs, regardless the type, set to default batch of 32
                    batch_size = 32

                try:
                    stop = args[5]
                    if stop.lower() == "y":
                        stop_word = True
                    else:
                        stop_word = False
                except IndexError:
                    # default to no stopword removal
                    stop_word = False

            except FileNotFoundError:
                print("Training file not found.")
            except ValueError:
                print("You must enter a valid number for the amount of epochs you'd like to run")
            except IndexError:
                print("You must enter all necessary arguments")

            if file_name is None or model_name is None or epochs is None:
                # Error message if args are missing
                print("You must enter all arguments for commands")
            else:
                # Train the model
                folds = 5
                print(
                    f"Training and testing new model named {model_name} for {epochs} epochs, with a {batch_size} batch size.")
                print(f"{model_name} will be using {folds} fold cross validation.")
                print(f"Stopword removal is {'enabled' if stop_word else 'disabled'}.")
                train_model(file_name, model_name, epochs, batch_size, stopword=stop_word, folds=folds)
        elif args[0].lower() == "wordcount":
            file_name = args[1]
            analyze_file_wordcounts(file_name)
            print("Analysis has completed")
        elif args[0].lower() == "check":
            training_file = args[1]
            predictions_file = args[2]
            trim = args[2]

            if trim.lower() == 'y':
                trim_desc = True
            else:
                trim_desc = False

            analyze_predictions(training_file, predictions_file, trim_descs=trim_desc, create_new_file=False)
            print(f"Checked and exported the file to {predictions_file.replace('.csv', '_checked.csv')}")
        elif args[0].lower() == "preproc":
            file = args[1]
            try:
                trim_data(file, True)
            except FileNotFoundError:
                print("You must enter a valid file name!")
        elif args[0].lower() == "quit":
            break


if __name__ == "__main__":
    main()
