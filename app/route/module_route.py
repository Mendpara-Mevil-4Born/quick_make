from flask import Blueprint, render_template
from app.controllers.module_controller import pruposal_module, module_details_api

pruposal_blueprint = Blueprint('pruposal_routes', __name__)

# Corrected POST method handling using @pruposal_blueprint.route
# @pruposal_blueprint.route('/pruposal', methods=['POST'])
# def pruposal_module_view():
#     return pruposal_module()

# @pruposal_blueprint.route('/pruposal-page', methods=['GET'])
# def sum_page():
#     return render_template('pruposal_form.html')
# pruposal_blueprint.add_url_rule('/module', view_func=pruposal_module, methods=['POST'])  # Endpoint: /v2/summarize
pruposal_blueprint.add_url_rule('/ai-powered', view_func=module_details_api, methods=['POST'])  # Endpoint: /v2/summarize

@pruposal_blueprint.route('/ai-powered', methods=['GET'])
def sum_page():
    return render_template('module_form.html')
