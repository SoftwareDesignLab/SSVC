import os
from flask import make_response, jsonify, request
from flask_restful import Resource
from ssvc_backend.mission_wellbeing.mission_and_wellbeing_backend import get_mission_and_wellbeing_impact


class MissionWellbeingApi(Resource):
    def get(self):
        """
        Method to get CISA value of a vulnerability based off of users self reported
        mission and well being from users
        :return:
        """
        query = request.args.to_dict()
        mission = query['mission'] if 'mission' in query else None
        well_being = query['wellbeing'] if 'wellbeing' in query else None

        if mission == "" or well_being == "" or mission is None or well_being is None:
            response_data = {
                "message": "There was an error with your request. Please ensure you supply both a mission and wellbeing value."
            }
            response = make_response(jsonify(response_data), 400)
            response.headers["Content-Type"] = "application/json"
            return response

        value = get_mission_and_wellbeing_impact(mission, well_being)

        if value is not None:
            # respond normally
            response_data = {
                "missionAndWellbeing": value
            }
        else:
            # if the status is None, an error occured, send a message to the user
            response_data = {
                "message": "There is an error with your request, please ensure you entered proper values for mission and well being."
            }
        response = make_response(jsonify(response_data), 200 if value is not None else 400)
        response.headers["Content-Type"] = "application/json"
        return response
