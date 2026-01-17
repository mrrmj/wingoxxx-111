import requests
import json
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from config import WINGO_API_URL, DATABASE_URL
Base = declarative_base()

class WinGoResult(Base):
    __tablename__ = "wingo_results"
    id = Column(Integer, primary_key=True)
    issueNumber = Column(String, unique=True, nullable=False)
    color = Column(String)
    size = Column(String)
    openResult = Column(Integer)
    openTime = Column(DateTime)

    def __repr__(self):
        return f"<WinGoResult(issueNumber=\\'{self.issueNumber}\\', color=\\'{self.color}\\', size=\\'{self.size}\\')>"

def get_latest_results():
    try:
        response = requests.get(WINGO_API_URL)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        return data.get("data", [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from WinGo API: {e}")
        return []

def store_results(results):
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    for result in results:
        issue_number = result.get("issueNumber")
        existing_result = session.query(WinGoResult).filter_by(issueNumber=issue_number).first()
        if not existing_result:
            open_time_str = result.get("openTime")
            try:
                # Convert timestamp (milliseconds) to datetime object
                open_time = datetime.fromtimestamp(open_time_str / 1000) if open_time_str else None
            except TypeError:
                open_time = None

            new_result = WinGoResult(
                issueNumber=issue_number,
                color=result.get("color"),
                size=result.get("size"),
                openResult=result.get("openResult"),
                openTime=open_time
            )
            session.add(new_result)
            print(f"Stored new result: {issue_number}")
        else:
            print(f"Result {issue_number} already exists.")
    session.commit()
    session.close()

def get_historical_data(limit=500):
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    data = session.query(WinGoResult).order_by(WinGoResult.openTime.desc()).limit(limit).all()
    session.close()
    return data

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True)
    issueNumber = Column(String, nullable=False)
    predicted_color = Column(String)
    predicted_size = Column(String)
    confidence_score = Column(Integer)
    actual_color = Column(String)
    actual_size = Column(String)
    timestamp = Column(DateTime, default=datetime.now)

    def __repr__(self):
        return f"<PredictionLog(issueNumber=\\'{self.issueNumber}\\', predicted_color=\\'{self.predicted_color}\\')>"

def store_prediction(issue_number, predicted_color, predicted_size, confidence_score, actual_color=None, actual_size=None):
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    
    new_log = PredictionLog(
        issueNumber=issue_number,
        predicted_color=predicted_color,
        predicted_size=predicted_size,
        confidence_score=confidence_score,
        actual_color=actual_color,
        actual_size=actual_size
    )
    session.add(new_log)
    session.commit()
    session.close()
    print(f"Stored prediction for {issue_number}")

def get_prediction_history(limit=10):
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()
    history = session.query(PredictionLog).order_by(PredictionLog.timestamp.desc()).limit(limit).all()
    session.close()
    return history

if __name__ == "__main__":
    # Example usage (for testing purposes)
    # You would typically call these functions from your main bot logic
    latest_data = get_latest_results()
    if latest_data:
        store_results(latest_data)
    
    historical_data = get_historical_data(10)
    for record in historical_data:
        print(record)


