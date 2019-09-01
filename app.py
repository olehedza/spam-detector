from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from flask_cors import CORS

import uuid
import jwt
from werkzeug.security import generate_password_hash, check_password_hash

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

import os
from datetime import datetime, timedelta
from functools import wraps

from count_vectorizer import predict


# init app
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Database setup
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + \
    os.path.join(BASE_DIR, 'db.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'secret_key'

# Init db
db = SQLAlchemy(app)

# Init marshmallow
ma = Marshmallow(app)

# User model


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    public_id = db.Column(db.String(50), unique=True)
    name = db.Column(db.String(50))
    password = db.Column(db.String(80))
    admin = db.Column(db.Boolean)


# User Schema
class UserSchema(ma.Schema):
    class Meta:
        fields = ('public_id', 'name', 'password', 'admin')


# init user schema
user_schema = UserSchema()
users_schema = UserSchema(many=True)


# Message model
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(500), nullable=False)
    spam = db.Column(db.Boolean)

    user_id = db.Column(db.Integer, db.ForeignKey(
        'user.id'), nullable=False)
    user = db.relationship(
        'User', backref=db.backref('messages', lazy=True))

    def __repr__(self):
        return f'<Message {self.pk}>'


# Message Schema
class MessageSchema(ma.Schema):
    class Meta:
        fields = ('id', 'text', 'spam')


# Init schema
message_schema = MessageSchema()
messages_schema = MessageSchema(many=True)


# Decorator for token required
def token_required(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        token = None

        if 'x-access-token' in request.headers:
            token = request.headers['x-access-token']

        if not token:
            return jsonify({'message': 'Token is missing'}), 401

        try:
            data = jwt.decode(token, app.config['SECRET_KEY'])
            current_user = User.query.filter_by(
                public_id=data['public_id']).first()
        except:
            return jsonify({'message': 'Token is invalid'}), 401

        return func(current_user, *args, **kwargs)

    return decorated


# ROUTES

# Get all users
@app.route('/user', methods=['GET'])
@token_required
def get_all_users(current_user):
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'})

    all_users = User.query.all()
    result = users_schema.dump(all_users)

    return jsonify({'users': result})


# Get user
@app.route('/user/<public_id>', methods=['GET'])
@token_required
def get_user(current_user, public_id):
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'})

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({'message': 'No user found!'})

    result = user_schema.dump(user)

    return jsonify({'user': result})


# Create user
@app.route('/user', methods=['POST'])
@token_required
def create_user(current_user):
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'})

    data = request.get_json()
    hashed_password = generate_password_hash(data['password'], method='sha256')
    new_user = User(
        public_id=str(uuid.uuid4()),
        name=data['name'],
        password=hashed_password,
        admin=False
    )

    db.session.add(new_user)
    db.session.commit()

    return jsonify({'message': 'New user created!'})


# Get all users
@app.route('/user/<public_id>', methods=['PUT'])
@token_required
def promote_user(current_user, public_id):
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'})

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({'message': 'No user found!'})

    user.admin = True
    db.session.commit()

    return jsonify({'message': 'User has been promoted.'})


# Delete user
@app.route('/user/<public_id>', methods=['DELETE'])
@token_required
def delete_user(current_user, public_id):
    if not current_user.admin:
        return jsonify({'message': 'Permission denied!'})

    user = User.query.filter_by(public_id=public_id).first()

    if not user:
        return jsonify({'message': 'No user found!'})

    db.session.delete(user)
    db.session.commit()

    return jsonify({'message': 'User has been deleted.'})


# Login
@app.route('/login')
def login():
    auth = request.authorization

    if not auth or not auth.username or not auth.password:
        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})

    user = User.query.filter_by(name=auth.username).first()

    if not user:
        return make_response('Could not verify', 401, {'WWW-Authenticate': 'Basic realm="Login required!"'})

    if check_password_hash(user.password, auth.password):
        token = jwt.encode({'public_id': user.public_id,
                            'exp': datetime.utcnow() + timedelta(minutes=30)}, app.config['SECRET_KEY'])

        return jsonify({'token': token.decode('UTF-8')})

    return make_response('Could not verify', 401, {
        'WWW-Authenticate': 'Basic realm="Login required!"'})


# Register user
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    user = User.query.filter_by(name=data['username']).first()

    if user:
        return jsonify({'message': 'User already exist'}), 422

    hashed_password = generate_password_hash(data['password'], method='sha256')
    new_user = User(
        public_id=str(uuid.uuid4()),
        name=data['username'],
        password=hashed_password,
        admin=False
    )

    db.session.add(new_user)
    db.session.commit()

    token = jwt.encode({'public_id': new_user.public_id,
                        'exp': datetime.utcnow() + timedelta(minutes=30)}, app.config['SECRET_KEY'])

    return jsonify({'success': True, 'token': token.decode('UTF-8')})


# Get all messages
@app.route('/message', methods=['GET'])
@token_required
def get_all_messages(current_user):

    messages = Message.query.filter_by(user_id=current_user.id).all()

    result = messages_schema.dump(messages)

    return jsonify({'messages': result})


# Check message on spam
@app.route('/message', methods=['POST'])
@token_required
def spam_detect(current_user):
    cv = predict()

    # Naive Bayes Classifier
    NB_spam_model = open('NB_spam_model.pkl', 'rb')
    clf = joblib.load(NB_spam_model)

    message = request.get_json()
    data = [message['text']]

    vect = cv.transform(data).toarray()
    my_prediction = bool(clf.predict(vect))

    new_message = Message(
        text=message['text'], spam=my_prediction, user_id=current_user.id)
    db.session.add(new_message)
    db.session.commit()

    result = message_schema.dump(new_message)

    return jsonify({'message': 'Text processed', 'text': message['text'], 'spam': my_prediction})


@app.route('/message/<message_id>', methods=['DELETE'])
@token_required
def delete_message(current_user, message_id):
    message = Message.query.filter_by(
        id=message_id, user_id=current_user.id).first()

    if not message:
        return jsonify({'message': 'No message found!'})

    db.session.delete(message)
    db.session.commit()

    return jsonify({'message': 'Message item deleted'})


# server running point
if __name__ == "__main__":
    app.run(debug=True)
