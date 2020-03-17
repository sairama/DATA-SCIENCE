from flask import Flask,render_template

app = Flask(__name__)

@app.route('/')
def index():
    #puppies = ['Fluffy','Rufus', 'Spike']
    user_logged_in = False
    return render_template('contrlflow.html',user_logged_in =user_logged_in)


if __name__ == '__main__':
    app.run(debug=True)
