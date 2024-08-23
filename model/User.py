from sqlalchemy import Column, Integer, String

from database import Base


class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(50), unique=True, nullable=False)
    password = Column(String(50), nullable=False)
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password
