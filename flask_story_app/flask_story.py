from flask import Flask, render_template, request, abort, jsonify
import flask
from flask import request
import numpy as np
import pickle
from server import make_story

my_story_model = pickle.load(open("hans_model.pkl", "rb"))

app = flask.Flask(__name__)

post = {
            'author': 'Michael Weber',
            'title': 'Story Time'
        }




@app.route("/")
@app.route("/home")
def home():
    return flask.render_template('home.html', post=post)

@app.route("/write", methods=['POST'])
def display_story():
    initial_text, generated_text = make_story()
    return flask.render_template('story.html',
        initial_text=initial_text,
        story=generated_text,
        post=post)



if __name__ == '__main__':
    app.run(debug=True)
