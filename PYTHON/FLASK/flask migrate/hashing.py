from flask_bcrypt import Bcrypt
bcrypt = Bcrypt()

password = 'supersecretpassword'

hashed_password =bcrypt.generate_password_hash(password = password)

print(hashed_password)
