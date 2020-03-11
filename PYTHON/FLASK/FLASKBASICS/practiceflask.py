from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/')
def index():
    return "<h1> Go to /puppy_name/name nd see the result </h1>"

@app.route('/puppy_name/<name>')
def puppylatin(name):

    pupname = ''

    if name[-1] =='y':
        pupname = name[:-1] + 'iful'
    else:
        pupname = name + 'y'

    return "<h1> your puppy latin nme is {}".format(pupname)

if __name__ == '__main__':
    app.run()
