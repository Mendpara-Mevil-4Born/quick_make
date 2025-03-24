from flask import Blueprint
from app.route.module_route import pruposal_blueprint
from app.route.summarized_route import summ_blueprint
main_blueprint = Blueprint('main_blueprint', __name__)
main_blueprint.register_blueprint(pruposal_blueprint, url_prefix="/module")  # Nested under /v1/upload

# sum_blueprint = Blueprint('sum_blueprint', __name__)
# sum_blueprint.register_blueprint(summ_blueprint, url_prefix="/upload_pruposal")  # Nested under /v2/upload

v2_blueprint = Blueprint('v2_blueprint', __name__)

# Register the `summ_blueprint` under `/v2/upload_pruposal`
v2_blueprint.register_blueprint(summ_blueprint, url_prefix="/upload_pruposal")