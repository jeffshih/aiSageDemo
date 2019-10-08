from flask import Flask 
from flask_sqlalchemy import SQLAlchemy 
db = SQLAlchemy()

def create_App():
    app = Flask(__name__, instance_relative_config=False)
    db.init_app(app)
    app
