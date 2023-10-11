import os
from flask import make_response, jsonify
from flask_restful import Resource
from ssvc_backend.mission_wellbeing.mission_and_wellbeing_backend import get_mission_and_wellbeing_impact


class MissionWellbeingApi(Resource):
    def get(self, mission, well_being):
        """
        Method to get CISA value of a vulnerability based off of users self reported
        mission and well being from users
        :return:
        """

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
