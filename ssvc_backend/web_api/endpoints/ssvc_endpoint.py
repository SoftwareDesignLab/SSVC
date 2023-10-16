import os
from pathlib import Path
from flask import make_response, jsonify, request
from flask_restful import Resource
from ssvc_backend.ssvc_decision_backend import get_ssvc_score, get_score_api_bypass


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

        # if cve_id is not specified, return an error
        if cve_id is None or cve_id == "":
            response_data = {
                "message": "There was an error with your request. Please ensure you inputted a valid CVE ID and proper delay with requests."
            }
            response = make_response(jsonify(response_data), 400)
            response.headers["Content-Type"] = "application/json"
            return response

        # Collect ssvc score
        if mission is not None and well_being is not None:
            score, data = get_ssvc_score(cve_id, mission, well_being, description, exploit_status=exploit_status)
        else:
            # if the mission and wellbeing was not given, start with requesting the information normally but for low
            # and then request the score by the way of bypassing the exploit status NVD request since exploit status
            # will be correct and accurate (Allows for test of ACTIVE exploitation) before continuing without verifying.
            score, data = get_ssvc_score(cve_id, "MINIMAL", "MINIMAL", description=description,
                                         exploit_status=exploit_status)
            exploit_status = data[0]
            # get decisions based on medium and high values of mission and well being impact
            med_score, med_data = get_score_api_bypass(cve_id, description, "MEDIUM", exploit_status)
            high_score, high_data = get_score_api_bypass(cve_id, description, "HIGH", exploit_status)
        # package respones into dictionary to convert into json
        if score is not None:
            # respond normally with a CVE ID, It's SSVC Score, and information regarding decisions
            response_data = {
                "cveId": cve_id,
                "exploitStatus": data[0],
                "automatable": data[1],
                "technicalImpact": data[2],
            }

            # append data based on specivity of api request
            if mission is not None and well_being is not None:
                response_data["ssvcScore"] = score
                response_data["missionAndWellbeing"] = data[3]
            else:
                # if mission and well being are not specified, return based on each value of mission
                # and well being
                response_data["ssvcScoreLow"] = score
                response_data["ssvcScoreMedium"] = med_score
                response_data["ssvcScoreHigh"] = high_score

        else:
            # if the status is None, an error occured, send a message to the user
            response_data = {
                "cveId": cve_id,
                "message": "There was an error with your request. Please ensure you inputted a valid CVE ID and proper delay with requests."
            }
        response = make_response(jsonify(response_data), 200 if score is not None else 400)
        response.headers["Content-Type"] = "application/json"
        return response
