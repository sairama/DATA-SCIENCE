from flask import Flask,render_template,request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('formindex.html')

@app.route('/signup_form')
def formsignup():
    return render_template('formsignup.html')

@app.route('/thankyou')
def thanku():
    first=request.args.get('first')
    last=request.args.get('last')
    return render_template('thanku.html',first=first,last=last)

@app.errorhandler(404)
def page_not_found(e):
    return render_template('form404.html') ,404



if __name__ == '__main__':
   app.run(debug = True)
