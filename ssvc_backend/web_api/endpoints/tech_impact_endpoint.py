import os
from pathlib import Path
from flask import make_response, jsonify, request
from flask_restful import Resource
from ssvc_backend.tech_impact.tech_impact_backend import get_tech_impact

class TechImpactApi(Resource):
    """
    Flask endpoint for technical impact to interact with the various apis and resources available to
    generate and return information relating to the CVE.
    """
    def get(self):
        """
        Method to determine whether a CVE's technical impact if exploited
        :return:
        """

        query = request.args.to_dict()
        cve_id = query['cveId'] if 'cveId' in query else None
        description = query['description'] if 'description' in query else None

        # if description is not specified, set to None to auto-retrieve a description
        if description == "":
            description = None

        # if cve_id is not specified, return an error
        if cve_id is None or cve_id == "":
            response_data = {
                "message": "There was an error with your request. Please ensure you inputted a valid CVE ID and proper delay with requests."
            }
            response = make_response(jsonify(response_data), 400)
            response.headers["Content-Type"] = "application/json"
            return response

        impact = get_tech_impact(cve_id, print_status=False, description=description)

        # package respones into dictionary to convert into json
        if impact is not None:
            # respond normally with a CVE ID and it's technical impact
            response_data = {
                "cveId": cve_id,
                "tech_impact": impact,
            }
        else:
            # if the status is None, an error occured, send a message to the user
            response_data = {
                "cveId": cve_id,
                "message": "There was an error with your request. Please ensure you inputted a valid CVE ID and proper delay with requests."
            }
        response = make_response(jsonify(response_data), 200 if impact is not None else 400)
        response.headers["Content-Type"] = "application/json"
        return response
