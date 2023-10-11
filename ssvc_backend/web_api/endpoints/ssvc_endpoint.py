import os
from pathlib import Path
from flask import make_response, jsonify, request
from flask_restful import Resource
from ssvc_backend.ssvc_decision_backend import get_ssvc_score


class SsvcScoringApi(Resource):
    """
    Flask endpoint for SSVC scoring to interact with the various apis and resources available to
    generate and return SSVC score relating to the CVE.
    """

    def get(self):
        """
        Get endpoint for ssvc score
        :return:
        """
        query = request.args.to_dict()
        cve_id = query['cveId'] if 'cveId' in query else None
        mission = query['mission'] if 'mission' in query else None
        well_being = query['wellbeing'] if 'wellbeing' in query else None
        description = query['description'] if 'description' in query else None
        exploit_status = query['exploitStatus'] if 'exploitStatus' in query else None

        # Collect ssvc score
        if mission is not None and well_being is not None:
            score, data = get_ssvc_score(cve_id, mission, well_being, description, exploit_status=exploit_status)

        # package respones into dictionary to convert into json
        if score is not None:
            # respond normally with a CVE ID, It's SSVC Score, and information regarding decisions
            response_data = {
                "cveId": cve_id,
                "ssvcScore": score,
                "exploitStatus": data[0],
                "automatable": data[1],
                "technicalImpact": data[2],
                "missionAndWellbeing": data[3]
            }
        else:
            # if the status is None, an error occured, send a message to the user
            response_data = {
                "cveId": cve_id,
                "message": "There was an error with your request. Please ensure you inputted a valid CVE ID and proper delay with requests."
            }
        response = make_response(jsonify(response_data), 200 if score is not None else 400)
        response.headers["Content-Type"] = "application/json"
        return response
