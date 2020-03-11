from flask import Flask

app = Flask(__name__)

@app.route('/') #127.0.0.1:5000
def index():
    return " <h1> hello puppy </h1> "

@app.route('/information') 
def info():
    return "<h1> puppies are cute </h1>"

if __name__ == "__main__":
    app.run()
