import os
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import re
from keras.callbacks import CSVLogger
from datetime import datetime, time
import numpy as np
import statistics
import nltk
from keras.optimizers import Adam
from nltk.corpus import stopwords
# noinspection PyUnresolvedReferences
import tensorflow_text as text  # do not remove this import, it is required for hub.load()
from sklearn.model_selection import KFold

PREPROCESS_SEQ_LENGTH = 128

# Modes is a global dictionary of each mode that the classifier can have, and their assigned numeric -> display values
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

SELECTED_MODE_NAME = "TECHNICAL_IMPACT"
SELECTED_MODE_DICT = MODES[SELECTED_MODE_NAME]


def update_classifier_mode(new_mode):
    """
    Function that updates the mode of the classiifer
    :param new_mode: mode being switched to
    :return: Boolean of success
    """
    if str(new_mode).upper() not in MODES:
        print(f"That mode does not exist, valid modes are: {', '.join(MODES.keys())}")
        return False

    global SELECTED_MODE_NAME
    SELECTED_MODE_NAME = str(new_mode).upper()
    global SELECTED_MODE_DICT
    SELECTED_MODE_DICT = MODES[SELECTED_MODE_NAME]
    print(f"The model's mode has been updated to {new_mode.upper()}")
    return True


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
    layer = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(layer)

    model = tf.keras.Model(inputs=text_input, outputs=[layer])
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

    model.compile(optimizer=Adam(learning_rate=.001),
                  loss='binary_crossentropy',
                  metrics=metrics)

    return model


def train_model(file_name, training_name, epochs, batch_size=32, downsample=True, stopword=False, folds=5):
    """
    Function that trains a new model using a bert classifier from intake of a csv file
    with the format of CVE Descriptions [Text] (C I Description) and a Decision value
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

    # Split into dataframes for each decision value
    df_0 = df[df['Decision'] == SELECTED_MODE_DICT[0]]
    df_1 = df[df['Decision'] == SELECTED_MODE_DICT[1]]

    # Downsampling larger sample size to match smaller amounts
    if downsample:
        #
        if df_1.shape[0] > df_0.shape[0]:
            df_1 = df_1.sample(n=df_0.shape[0])
            print(f"Downsampling is enabled, the training will be downsampled to {SELECTED_MODE_DICT[0]}'s size with {df_0.shape[0]} samples each.")
        else:
            df_0 = df_0.sample(n=df_1.shape[0])
            print(f"Downsampling is enabled, the training will be downsampled to {SELECTED_MODE_DICT[1]}'s size with {df_1.shape[0]} samples each.")


    # Create unified balanced dataframe
    df_balanced = pd.concat([df_1, df_0])
    df_balanced = df_balanced.reset_index(drop=True)
    df_balanced['total'] = df_balanced['Decision'].apply(lambda x: 1 if str(x).lower() == SELECTED_MODE_DICT[1].lower() else 0)

    inputs = df_balanced['Text']
    targets = df_balanced['total']

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
        y_predicted = y_predicted.flatten()

        # Set prediction to 1 if > confidence_interval
        confidence_interval = 0.5
        y_predicted = np.where(y_predicted > confidence_interval, 1, 0)

        # Save the model for this fold
        model.save(f"./models/{training_name}/fold_{fold_no}")
        # Create data obj for pandas dataframe
        data = []
        for i, cve in enumerate(in_test):
            # create decision string of based off the current model's mode, see global MODES variable's for specifics
            decision = SELECTED_MODE_DICT[1] if y_predicted[i] == 1 else SELECTED_MODE_DICT[0]
            data.append([cve, decision])

        # Create predictions data frame
        pdf = pd.DataFrame(data, columns=["description", "decision"])
        pdf.to_csv(f"./models/{training_name}/fold_{fold_no}/predictions.csv", index=False)
        print(f"(FOLD {fold_no}) CSV File for the predictions made in this training have bene exported.")
        analyze_predictions(file_name, f'models/{training_name}/fold_{fold_no}/predictions.csv', create_new_file=False,
                            trim_descs=stopword)
        check_positives(f'models/{training_name}/fold_{fold_no}/predictions.csv')
        # calc f1 score
        calculate_f1_score(f"./models/{training_name}/fold_{fold_no}/log.csv")
        # test predictions evaluation
        evaluated = model.evaluate(in_test, targets_test, batch_size=batch_size)
        fold_scores.append(evaluated)
        # assign statistics of prediction
        predictions_data = list_to_data_dict(evaluated)

        predictions_log = pd.DataFrame(columns=["loss", "accuracy", "auc", "precision", "recall", "f1_score"],
                                       data=predictions_data, index=[0])

        predictions_log.to_csv(f"models/{training_name}/fold_{fold_no}/predictions_log.csv", index=False)
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] (FOLD {fold_no}) The training and testing of {training_name} has completed")
        fold_no += 1

    print(f"[{datetime.now().strftime('%H:%M:%S')}] All folds for {training_name} have completed")
    avg_scores = average_fold_stats(fold_scores)
    avg_data = list_to_data_dict(avg_scores)
    predictions_log = pd.DataFrame(columns=["loss", "accuracy", "auc", "precision", "recall", "f1_score"],
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
    precision = stats_list[2]
    recall = stats_list[3]
    auc = stats_list[4]
    f1_score = safe_divide(2 * (precision * recall), (precision + recall))

    data = {
        "loss": loss,
        "accuracy": accuracy,
        "auc": auc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }

    return data


def print_help():
    """
    Function that prints out the commands for the script
    :return:
    """
    print("")
    print("SSVC Technical Impact classifier (BERT):")
    print_command("Train and Test",
                  "Trains and tests a new model from a csv file using user specified input and 5 fold cross validation.",
                  "tat <model_name> <train_file> <epochs> (batch_size: default 32) (stpoword: y/n)",
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
    print_command("Check Positives and Negatives", "Check the negatives and positives of a checked predictions file",
                  "positives <checked_predictions_file", "positives models/100_epochs/predictions.csv")
    print("")
    print_command("Predict Values", "Takes in a CSV file containing CVE descriptions, a model and fold number, and outputs predictions.",
                  "predict <model> <file> <fold> (confidence interval: 0.5 default)", "predict 100samples to_predict.csv 4 0.6")
    print("")
    print_command("Change Model Mode", "Changes the mode of the model.", "mode <mode>", "mode technical_impact")
    print("")
    print("Quit - terminates program")
    print("Help - displays this")
    print("")
    print(f"Current model mode: {SELECTED_MODE_NAME}")


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
        f1_score = 2 * (precision * recall) / (precision + recall)
        f1_scores.append(f1_score)

    # create new column
    df['f1_score'] = f1_scores

    # export new file
    df.to_csv(file)


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
    try:
        # make dir if it doesn't exist
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
    :param trim_descs: Whether the analysis trims descriptions or not
    :return:
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
    for index, row in predictions_file.iterrows():
        desc = row["description"]
        impact = row["decision"]

        search = training_file[(training_file["Text"] == desc)]
        train_data_impact = search.iloc[0]['Decision']

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


def get_predictions_list(text, predicted_values, include_desc_in_return = False, confidence_interval = 0.5):
    """
    This function is to take text that the model is predicting along with the numerical values that the model
    had predicted, and then it will return a list of the predictions in plain text.
    :param text: List of values that the model is predicting
    :param predicted_values: List of values that the model returned
    :param include_desc_in_return: Boolean value dictating whether the list returned contains the description as well.
    :param confidence_interval: Confidence interval that determines if it should be a 1 instead of a 0.
    :return: list of string predictions
    """
    predicted_values = np.where(predicted_values > confidence_interval, 1, 0)

    # Create data obj for pandas dataframe
    data = []
    for i, cve in enumerate(text):
        # create string of decision based off modes settings (see MODES global for specifics)
        decision = SELECTED_MODE_DICT[1] if predicted_values[i] == 1 else SELECTED_MODE_DICT[0]
        if include_desc_in_return:
            data.append([cve, decision])
        else:
            data.append(decision)

    return data


def check_positives(checked_predictions_file):
    """
    Creates and exports a file that contains true positives, true negatives etc.
    based off a checked predictions file
    Final score includes accuracy, precision, recall, f1 score
    :param checked_predictions_file: predictions file
    :return:
    """
    predictions = pd.read_csv(checked_predictions_file)

    tp = 0  # true positive - correct predicted as 1
    tn = 0  # true negative - correct predicted as 0
    fp = 0  # false positive - predicted as 1 should be 0
    fn = 0  # false negative - 0 should be 0

    for index, row in predictions.iterrows():
        decision = row['decision']
        correct = row['correct']

        if decision == SELECTED_MODE_DICT[1] and correct == "correct":
            # True positive
            tp += 1
        elif decision == SELECTED_MODE_DICT[0] and correct == "correct":
            # True negative
            tn += 1
        elif decision == SELECTED_MODE_DICT[1] and correct == "incorrect":
            # false positive
            fp += 1
        elif decision == SELECTED_MODE_DICT[0] and correct == "incorrect":
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


def predict_data(model_name, data_file, fold=1, confidence_interval=0.5):
    """
    The predict data function is a function that will load an already trained model, and open and process a csv
    file with CVE descriptions under a header of "Text", and update the csv file with a new "prediction" column with the
    prediction for that specific cve.
    :param model_name: name of a trained model in the 'models' folder
    :param data_file: csv file with the data you want predicted
    :return:
    """

    # get and load weights to the model
    model = get_bert_model()
    model.load_weights(f'models/{model_name}/fold_{fold}')

    df = pd.read_csv(data_file)
    input = df['Text']

    predictions = model.predict(input)
    predictions = predictions.flatten()
    # Set prediction to 1 if > confidence_interval
    predictions = np.where(predictions > confidence_interval, 1, 0)
    data = []

    for i, cve in enumerate(input):
        # create string of decision based off modes settings (see MODES global for specifics)
        decision = SELECTED_MODE_DICT[1] if predictions[i] == 1 else SELECTED_MODE_DICT[0]
        data.append(decision)  # add to list of predictions to add to df.

    df['prediction'] = data
    d = datetime.now().strftime("%m-%d-%Y_%H_%M_%S")
    # Make predictions folder to house predictions along with info regarding the predictions
    safe_mk_dir(f"predictions/prediction_{d}")

    df.to_csv(f"predictions/prediction_{d}/predictions.csv", index=False)

    # Create a text file with predictions
    with open(f"predictions/prediction_{d}/prediction_information.txt", "w+") as info:
        lines = [f"Model Used: {model_name}\n",
                 f"Weights from fold: {fold}\n",
                 f"Confidence interval for predictions: {confidence_interval}\n",
                 f"Predictions made on: {d}"]

        info.writelines(lines)


def main():
    """
    Main function that runs the program
    :return:
    """

    # housekeeping checks (makes sure all proper folders are made)
    safe_mk_dir(f"./models")
    safe_mk_dir(f"./preprocessed")
    safe_mk_dir(f"./data_analysis")
    safe_mk_dir("./predictions")

    # print help on start
    print_help()
    while True:
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
        elif args[0].lower() == "positives":
            file = args[1]
            try:
                check_positives(file)
                print("Exported checked file containing true positives,true negatives, false positives, false negatives")
            except FileNotFoundError:
                print("You must enter a valid file name!")
        elif args[0].lower() == "predict":
            if len(args) <= 4:
                print("You must enter the correct number of minimal arguments for this command.")
            else:
                model = args[1]
                to_be_predicted = args[2]
                fold = int(args[3])
                if len(args) == 5:
                    confidence = float(args[4])
                else:
                    confidence = 0.5

                predict_data(model, to_be_predicted, fold, confidence)
        elif args[0].lower() == "mode":
            if len(args) != 2:
                print(f"A list of valid classifier modes are: {', '.join(MODES.keys())}")
            else:
                new_mode = args[1]
                update_classifier_mode(new_mode)
        elif args[0].lower() == "quit":
            break
        elif args[0].lower() == "help":
            print_help()


if __name__ == "__main__":
    main()