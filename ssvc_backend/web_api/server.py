from flask import Flask
from flask_restful import Resource, Api

from ssvc_backend.web_api.endpoints.exploit_status_endpoint import ExploitStatusApi
from ssvc_backend.web_api.endpoints.automatability_endpoint import AutomatabilityApi
app = Flask(__name__)

api = Api(app)

api.add_resource(ExploitStatusApi, '/exploitstatus/<cve_id>')
api.add_resource(AutomatabilityApi, '/automatability/<cve_id>')

if __name__ == '__main__':
    app.run(debug=True)