from flask import Flask
from flask_restful import Resource, Api

from ssvc_backend.web_api.endpoints.exploit_status_endpoint import ExploitStatusApi
from ssvc_backend.web_api.endpoints.automatability_endpoint import AutomatabilityApi
from ssvc_backend.web_api.endpoints.mission_wellbeing_endpoint import MissionWellbeingApi
from ssvc_backend.web_api.endpoints.ssvc_endpoint import SsvcScoringApi
from ssvc_backend.web_api.endpoints.quick_ssvc_endpoint import QuickSsvcScoringApi
from ssvc_backend.web_api.endpoints.tech_impact_endpoint import TechImpactApi

app = Flask(__name__)

api = Api(app)

api.add_resource(ExploitStatusApi, '/exploitstatus/<cve_id>')
api.add_resource(AutomatabilityApi, '/automatability/<cve_id>')
api.add_resource(MissionWellbeingApi, '/wellbeing/<mission>/<well_being>')
api.add_resource(TechImpactApi, '/techimpact/<cve_id>')
api.add_resource(SsvcScoringApi, "/ssvc")
api.add_resource(QuickSsvcScoringApi, "/quickssvc")

if __name__ == '__main__':
    app.run(debug=True)