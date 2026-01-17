from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timedelta

from config import DATABASE_URL
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    telegram_id = Column(Integer, unique=True, nullable=False)
    is_premium = Column(Boolean, default=False)
    premium_expiry = Column(DateTime, nullable=True)
    joined_date = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<User(telegram_id={self.telegram_id}, is_premium={self.is_premium})>"

def get_or_create_user(telegram_id):
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    user = session.query(User).filter_by(telegram_id=telegram_id).first()
    if not user:
        user = User(telegram_id=telegram_id)
        session.add(user)
        session.commit()
        print(f"Created new user: {telegram_id}")
    session.close()
    return user

def set_premium_status(telegram_id, is_premium, duration_days=30):
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    user = session.query(User).filter_by(telegram_id=telegram_id).first()
    if user:
        user.is_premium = is_premium
        if is_premium:
            user.premium_expiry = datetime.now() + timedelta(days=duration_days)
        else:
            user.premium_expiry = None
        session.commit()
        print(f"User {telegram_id} premium status set to {is_premium}")
    session.close()

def check_premium_status(telegram_id):
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    user = session.query(User).filter_by(telegram_id=telegram_id).first()
    is_premium = False
    if user and user.is_premium and user.premium_expiry and user.premium_expiry > datetime.now():
        is_premium = True
    session.close()
    return is_premium

if __name__ == "__main__":
    # Example usage
    user_id = 12345
    get_or_create_user(user_id)
    print(f"Is user {user_id} premium? {check_premium_status(user_id)}")
    set_premium_status(user_id, True)
    print(f"Is user {user_id} premium? {check_premium_status(user_id)}")
    set_premium_status(user_id, False)
    print(f"Is user {user_id} premium? {check_premium_status(user_id)}")


