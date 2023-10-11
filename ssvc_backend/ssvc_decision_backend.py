from ssvc_backend.automatability.automatability_backend import *
from ssvc_backend.exploit_status.exploit_status_backend import *
from ssvc_backend.mission_wellbeing.mission_and_wellbeing_backend import *
from ssvc_backend.tech_impact.tech_impact_backend import *

# SSVC Scoring Tree is a dictionary to correlate values to a SSVC CISA scoring
# Keys = Tuple value = SSVC Score (str)
# Tuple = (ExploitStatus (NONE, POC, ACTIVE), Automatable (True/False), Tech Impact (total/partial), Mission & Well Being (NONE/LOW/HIGH)
# Created off the tree from https://www.cisa.gov/ssvc-calculator
__SSVC_SCORING_TREE = {
    ("NONE", False, "partial", "LOW"): "TRACK",
    ("NONE", False, "partial", "MEDIUM"): "TRACK",
    ("NONE", False, "partial", "HIGH"): "TRACK",

    ("NONE", False, "total", "LOW"): "TRACK",
    ("NONE", False, "total", "MEDIUM"): "TRACK",
    ("NONE", False, "total", "HIGH"): "TRACK*",

    ("NONE", True, "partial", "LOW"): "TRACK",
    ("NONE", True, "partial", "MEDIUM"): "TRACK",
    ("NONE", True, "partial", "HIGH"): "ATTEND",

    ("NONE", True, "total", "LOW"): "TRACK",
    ("NONE", True, "total", "MEDIUM"): "TRACK",
    ("NONE", True, "total", "HIGH"): "ATTEND",

    ("POC", False, "partial", "LOW"): "TRACK",
    ("POC", False, "partial", "MEDIUM"): "TRACK",
    ("POC", False, "partial", "HIGH"): "TRACK*",

    ("POC", False, "total", "LOW"): "TRACK",
    ("POC", False, "total", "MEDIUM"): "TRACK*",
    ("POC", False, "total", "HIGH"): "ATTEND",

    ("POC", True, "partial", "LOW"): "TRACK",
    ("POC", True, "partial", "MEDIUM"): "TRACK",
    ("POC", True, "partial", "HIGH"): "ATTEND",

    ("POC", True, "total", "LOW"): "TRACK",
    ("POC", True, "total", "MEDIUM"): "TRACK*",
    ("POC", True, "total", "HIGH"): "ATTEND",

    ("ACTIVE", False, "partial", "LOW"): "TRACK",
    ("ACTIVE", False, "partial", "MEDIUM"): "TRACK",
    ("ACTIVE", False, "partial", "HIGH"): "ATTEND",

    ("ACTIVE", False, "total", "LOW"): "TRACK",
    ("ACTIVE", False, "total", "MEDIUM"): "ATTEND",
    ("ACTIVE", False, "total", "HIGH"): "ACT",

    ("ACTIVE", True, "partial", "LOW"): "ATTEND",
    ("ACTIVE", True, "partial", "MEDIUM"): "ATTEND",
    ("ACTIVE", True, "partial", "HIGH"): "ACT",

    ("ACTIVE", True, "total", "LOW"): "ATTEND",
    ("ACTIVE", True, "total", "MEDIUM"): "ACT",
    ("ACTIVE", True, "total", "HIGH"): "ACT"
}

# Decision cache for requests to speed up returns
# Keys = (cve_id, mission, wellbeing) -- case should be forced to maintain consistency
# Value = tuple of (ssvc score, (data associated))
__DECISION_CACHE = dict()

def get_ssvc_score(cve_id: str, mission: str, well_being: str, description=None, exploit_status=None):
    """
    A function that will return SSVC score decision based off of CVE-ID and user reported
    impact on mission and well being, and run auto scoring on the CVE.
    @param cve_id: CVE-ID of vulnerability
    @param mission: impact on mission
    @param well_being: impact on well being
    @param description: description of CVE-ID (this is optional, and good if the NVD has not published a CVE)
    @param exploit_status: exploit status (this is optional, if the exploit status is given, but not able
    @return: SSVC Decision along with the data associated with that decision (in tuple format: exploit status, automatability, tech impact, mission impact)
    """

    # Get description of CVE return None if desc is none
    if description is None:
        description = get_description(cve_id)
        if description is None:
            return None

    # Check the cached data in the script
    cache_key = (cve_id.upper(), mission.upper(), well_being.upper())
    if cache_key in __DECISION_CACHE:
        cached_tuple = __DECISION_CACHE[cache_key]
        ssvc_score = cached_tuple[0]
        data_tuple = cached_tuple[1]

        return ssvc_score, data_tuple

    # Retrieve values if not in cache
    tech_impact = get_tech_impact(cve_id, description=description)
    auto = is_automatable(cve_id, description=description)

    exploit = get_exploit_status(cve_id)

    # If exploit status provided by user is a POC, but NVD does not have
    # record of CVE, or other exploit status checks aren't passed
    # update the exploit status to POC.
    # If the exploit status is ACTIVE, it will stay as active.
    if exploit_status is not None and exploit_status.upper() == "POC":
        if not exploit == "ACTIVE":
            exploit = "POC"

    mission_impact = get_mission_and_wellbeing_impact(mission, well_being)

    # Convert retrieved information into a tuple and check the tree in
    data = (exploit, auto, tech_impact, mission_impact)

    # If something went wrong, return None
    if data not in __SSVC_SCORING_TREE:
        return None

    # return decision and the data associated with the decision
    return __SSVC_SCORING_TREE[data], data


def get_score_api_bypass(cve_id, description, mwb_impact, exploit_status):
    """
    Determining SSVC score for a cve with information that is already given/provided.
    This is not a recommended method of determining the value without ensuring values are correct
    and data provided is accurate with appropriate CNAs, if you do not have this information or think
    exploit status or something may change, please use get_ssvc_score()
    @param cve_id: CVE id being scored
    @param description: Descrpition of CVE
    @param mwb_impact: the impact of mission and well being, either "LOW", "MEDIUM" or "HIGH.
    @param exploit_status: exploit status of cve
    @return: ssvc score or none if not found
    """
    # Retrieve values if not in cache
    tech_impact = get_tech_impact(cve_id, description=description)
    auto = is_automatable(cve_id, description=description)
    exploit = get_exploit_status(cve_id)

    # Convert retrieved information into a tuple and check the tree in
    data = (exploit, auto, tech_impact, mwb_impact)

    # If something went wrong, return None
    if data not in __SSVC_SCORING_TREE:
        return None

    # return decision and the data associated with the decision
    return __SSVC_SCORING_TREE[data], data

def main():
    while True:
        cve_id = input("enter cve_id: ")
        mission = input("Enter mission: ")
        well = input("Enter well being: ")

        score, data = get_ssvc_score(cve_id, mission, well)
        print(score)
        print(data)

if __name__ == "__main__":
    main()