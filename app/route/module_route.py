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
@pruposal_blueprint.route('/ai-powered', methods=['POST'])
def ai_powered_api():
    print("ai_powered_api function called!")
    try:
        result = module_details_api()
        print("module_details_api returned successfully")
        return result
    except Exception as e:
        print(f"Error in ai_powered_api: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"An error occurred: {str(e)}"}, 500

@pruposal_blueprint.route('/ai-powered', methods=['GET'])
def sum_page():
    return render_template('module_form.html')
