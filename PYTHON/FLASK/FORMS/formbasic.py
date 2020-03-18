from flask import Flask , render_template
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

app = Flask(__name__)

app.config['SECRET_KEY'] = 'mysecretkey'


class Infoform(FlaskForm):

    breed = StringField("what breed are you ?")
    submit =  SubmitField('submit')

@app.route('/',methods=['GET','POST'])
def index():

    breed = False

    form = Infoform()
    if form.validate_on_submit():
        breed =  form.breed.data
        form.breed.data = ''
    return render_template('formhome.html',form = form,breed= breed)

if __name__ == '__main__':
   app.run(debug = True)
