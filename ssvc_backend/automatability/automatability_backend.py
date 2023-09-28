import os
import requests
import pandas as pd
from requests import JSONDecodeError
import TechnicalImpact.BERTClassifier.binary_classifier as binary_classifier

# IMPORTANT NOTE: make sure you have openpyxl installed.

# global dictionary that will cache decisions to save time, and liimt repeated requests to API
# Key = CVE_ID Value = decision
__DECISION_CACHE = dict()

def get_description(cve_id):
    """
    A function that takes in a CVE ID and calls the NVD api and returns its description
    :param cve_id: id of the cve
    :return: list of cwes associated with a cve
    """

    # making request to get the information of the CVE from
    request = requests.get(f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}").json()
    vulnerabilities = request["vulnerabilities"]

    # The way that json.loads() loads the cves from the NIST json response
    # is all id'd by "CVE" into a dictionary of information.
    if len(vulnerabilities) == 0:
        return None

    cve_dict = vulnerabilities[0]
    # The way json
    cve = cve_dict["cve"]

    # The description of the vulnerability is in a list of json objects.
    # english is always the first description object in the list.
    # so i will be abusing that to assign to variable
    description = cve["descriptions"][0]["value"]
    return description

def is_automatable(cve_id, working_dir=os.path.dirname(__file__), print_status=True):
    """
    function to output the status of the cve provided. If the CVE is in kev it will output in KEV
    otherwise, the code will find the cwes associated with the cve
    :param cve_id:
    :param working_dir: Directory where the ML model is looking for
    :return: boolean of automatability
    """
    error_msg = "There was an error with your request, please ensure you inputted a valid CVE ID, and have not been flooding requests."

    # check cache before doing anything
    global __DECISION_CACHE
    if cve_id in __DECISION_CACHE:
        return __DECISION_CACHE[cve_id]

    try:
        desc = get_description(cve_id)

        if desc is None:
            # input cve into cache before returning
            # if the description is none there is not a CVE with that ID
            # there would be an error if it was rate limiting
            __DECISION_CACHE[cve_id] = None
            if print_status:
                print(error_msg)
            return None

        if print_status:
            print(f"Processing description: {desc}")

        # Update the classifier's mode to automatability
        binary_classifier.update_classifier_mode("AUTOMATABILITY")
        # Call the BERT model to determine its automatability
        automatable = binary_classifier.predict_description(model_name="auto_binary_Adam_0015LR",
                                                            working_dir=working_dir,
                                                            cve_description=desc,
                                                            fold=5)

        # print automatability
        if print_status:
            print(f"{cve_id} is automatable." if automatable == "YES" else f"{cve_id} is NOT automatable.")

        if automatable == "YES":
            # cache decision value and return
            __DECISION_CACHE[cve_id] = True
            return True
        else:
            # cache decision value and return
            __DECISION_CACHE[cve_id] = False
            return False

    except JSONDecodeError:
        if print_status:
            print(error_msg)
        return None

def main():
    id = input("enter cve id: ")
    is_automatable(id)


if __name__ == "__main__":
    main()