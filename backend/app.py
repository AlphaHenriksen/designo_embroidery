from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/get", methods=["GET"])
def get_articles():
    return jsonify({"Hello": "World!"})
