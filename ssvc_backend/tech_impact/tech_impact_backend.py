import os
import requests
import pandas as pd
from requests import JSONDecodeError
from ssvc_backend.utils.ssvc_utils import get_description
import TechnicalImpact.logreg_model as logreg_model

# global dictionary that will cache decisions to save time, and liimt repeated requests to API
# Key = CVE_ID Value = decision
__DECISION_CACHE = dict()


def get_tech_impact(cve_id, working_dir=os.path.dirname(__file__), print_status=True, description=None):
    """
    function to output the status of the cve provided. If the CVE is in kev it will output in KEV
    otherwise, the code will find the cwes associated with the cve
    :param cve_id:
    :param working_dir: Directory where the ML model is looking for
    :param print_status: whether the results will be printed to standard input
    :param description: this will skip the description reception step, and process the description passed through, it is
    recommended to do this when doing SSVC scoring of a CVE so you avoid rate limiting on NVD.
    :return: partial or total technical impact determination.
    """
    error_msg = "There was an error with your request, please ensure you inputted a valid CVE ID, and have not been flooding requests."

    # check cache before doing anything
    global __DECISION_CACHE
    if cve_id in __DECISION_CACHE:
        return __DECISION_CACHE[cve_id]

    try:
        if description is None:
            desc = get_description(cve_id)

            if desc is None:
                # input cve into cache before returning
                # if the description is none there is not a CVE with that ID
                # there would be an error if it was rate limiting
                __DECISION_CACHE[cve_id] = None
                if print_status:
                    print(error_msg)
                return None
        else:
            desc = description

        if print_status:
            print(f"[Technical Impact] Processing description: {desc}")

        # call prediction function
        impact = logreg_model.predict_description(desc, working_dir)

        # print technical impact
        if print_status:
            print(f"{cve_id}'s technical impact is {impact}.")

        __DECISION_CACHE[cve_id] = impact
        return impact

    except JSONDecodeError:
        if print_status:
            print(error_msg)
        return None


def main():
    id = input("enter cve id: ")
    get_tech_impact(id)


if __name__ == "__main__":
    main()
