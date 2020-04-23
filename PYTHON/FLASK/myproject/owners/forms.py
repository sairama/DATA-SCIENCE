#owners
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField

class AddForm(FlaskForm):
    name = StringField('Name of owner:')
    pup_id = IntegerField("id of puppy : ")
    submit = SubmitField("add owner: ")
