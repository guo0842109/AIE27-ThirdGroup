import torch
from flask import Flask, request, jsonify
import models
import transform

import time
from collections import OrderedDict

app = Flask(__name__)


@app.route('/')
def hello():
    return "hello world"


if __name__ == '__main__':
    app.run()
