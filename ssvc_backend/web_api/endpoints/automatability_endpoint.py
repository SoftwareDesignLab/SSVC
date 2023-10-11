import os
from pathlib import Path
from flask import make_response, jsonify
from flask_restful import Resource
from ssvc_backend.automatability.automatability_backend import is_automatable

class AutomatabilityApi(Resource):
    """
    Flask endpoint for Automatability to interact with the various apis and resources available to
    generate and return information relating to the CVE
    """
    def get(self, cve_id):
        """
        Method to determine whether a CVE's exploit is automatable
        :return:
        """

        is_auto = is_automatable(cve_id, print_status=False)

        # package respones into dictionary to convert into json
        if is_auto is not None:
            # respond normally with a cave id and whether it is automatable
            response_data = {
                "cveId": cve_id,
                "automatable": is_auto,
            }
        else:
            # if the status is None, an error occurred, send a message to the
            response_data = {
                "cveId": cve_id,
                "message": "There was an error with your request. Please ensure you inputted a valid CVE ID and proper delay with requests."
            }
        response = make_response(jsonify(response_data), 200 if is_auto is not None else 400)
        response.headers["Content-Type"] = "application/json"
        return response
