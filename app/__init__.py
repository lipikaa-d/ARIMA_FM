from flask import Flask

def create_app():
    app = Flask(__name__)
    app.secret_key = 'secret-key'  # required for flashing messages

    from app.routes import main
    app.register_blueprint(main)

    return app
