from json import JSONDecodeError

import requests

def get_json(cve_id):
    """
    Method to get the json response as a dictionary from the requests library
    this can be sent thorugh
    @param cve_id:
    @return:
    """
    try:
        request = requests.get(f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}").json()
    except JSONDecodeError:
        return None
    return request

def get_description(cve_id, json_data=None):
    """
    A function that takes in a CVE ID and calls the NVD api and returns its description
    :param cve_id: id of the cve
    :param json_data: json data that would be returned by a requests.get() call. It will avoid calling api if provided
    :return: list of cwes associated with a cve
    """

    # making request to get the information of the CVE from
    if json_data is None:
        try:
            request = requests.get(f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}").json()
        except JSONDecodeError:
            return None
    else:
        request = json_data

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
    status = cve["vulnStatus"]

    if status == "Rejected":
        return "REJECTED"

    description = cve["descriptions"][0]["value"]

    return description