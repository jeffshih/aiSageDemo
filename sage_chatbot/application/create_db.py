import sys 
import sqlalchemy 
from sqlalchemy.ext.declarative import declarative_base 


def create_db():
    Base = declarative_base()
    engine = sqlalchemy.create_engine('sqlite:///./db/chatbot.db')
    Base.metadata.create_all(engine)

