from flask import Blueprint, render_template
from app.controllers.summarized import summ

# Blueprint for summarized routes
summ_blueprint = Blueprint('summarized_routes', __name__)

# Add a URL rule for the summarized route
summ_blueprint.add_url_rule('/summarize', view_func=summ, methods=['POST']) 

@summ_blueprint.route('/summarize-page', methods=['GET'])
def sum_page():
    return render_template('summarize.html')
