from flask import Flask, render_template,session,redirect,url_for,flash
from flask_wtf import FlaskForm
from wtforms import (StringField,BooleanField,
                    RadioField,SelectField,
                     TextAreaField,SubmitField)



app = Flask(__name__)

app.config['SECRET_KEY'] = 'my_key'

class SimpleForm(FlaskForm):
    breed = StringField('What breed are you?')
    submit = SubmitField('Click me')

@app.route('/',methods = ['GET','POST'])
def index():

    form =  SimpleForm()

    if form.validate_on_submit():
        session['breed'] = form.breed.data
        flash(f"you just changes your breede to: {session['breed']})")
        flash("hello")

        return redirect(url_for('index'))

    return render_template('flashalerts.html', form = form)

if __name__=='__main__':
    app.run(debug=True)
