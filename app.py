# filepath: c:\Users\mevil\Downloads\4born\10-3-25(Quick make- new model integrate for prize and time field)\Quick-make - model integrate--new changes\app.py
from flask import Flask, render_template
from flask_cors import CORS
from app.route.routes import main_blueprint, v2_blueprint
from app.controllers.module_controller import module_blueprint  # Import the blueprint

app = Flask(__name__, template_folder="app/templates")
CORS(app, resources={r"/v1/*": {"origins": "*"}})

@app.route('/hello_world')
def hello_world():
    return 'Hello, World!'

app.register_blueprint(main_blueprint, url_prefix="/v1")
app.register_blueprint(v2_blueprint, url_prefix="/v2")
app.register_blueprint(module_blueprint, url_prefix="/api")  # Register the blueprint

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    # Run the Flask app
    app.run(host="0.0.0.0", port=5000, debug=True)

    # for production
    # app.run(host="0.0.0.0", port=3035, debug=False) 
