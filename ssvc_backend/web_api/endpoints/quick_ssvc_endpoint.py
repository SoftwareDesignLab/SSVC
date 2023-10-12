import os
from pathlib import Path
from flask import make_response, jsonify, request
from flask_restful import Resource
from ssvc_backend.ssvc_decision_backend import get_decision_from_tree


class QuickSsvcScoringApi(Resource):
    """
    Flask endpoint for quick SSVC decisions based on already established/provided values
    by the user to determine decisions for a vulnerability based on the mission mission and well being
    values
    """

    def get(self):
        query = request.args.to_dict()
        tech_impact = query['technicalImpact'] if 'technicalImpact' in query else None
        automatable = query['automatable'] if 'automatable' in query else None
        exploit_status = query['exploitStatus'] if 'exploitStatus' in query else None

        if tech_impact is None or automatable is None or exploit_status is None:
            response_data = {
                "message": "There was an error with your request. Please ensure you input valid information for each part of the decision tree."
            }
            response = make_response(jsonify(response_data), 400)
            response.headers["Content-Type"] = "application/json"
            return response

        low, med, high = get_decision_from_tree(tech_impact, bool(automatable), exploit_status)

        if low is not None:
            response_data = {
                "ssvcScoreLow": low,
                "ssvcScoreMedium": med,
                "ssvcScoreHigh": high
            }
        else:
            # if the status is None, an error occured, send a message to the user
            response_data = {
                "message": "There was an error with your request. Please ensure you input valid information for each part of the decision tree."
            }
        response = make_response(jsonify(response_data), 200 if low is not None else 400)
        response.headers["Content-Type"] = "application/json"
        return response
