"""SQLAlchemy database models for RoverOps"""
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, JSON, ForeignKey, Enum as SQLEnum, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

Base = declarative_base()


class Mission(Base):
    """Mission table - stores mission metadata"""
    __tablename__ = "missions"

    mission_id = Column(String(255), primary_key=True, index=True)
    goal = Column(Text, nullable=False)
    status = Column(String(50), nullable=False, index=True)  # pending, planning, executing, complete, aborted, error
    current_step = Column(Integer, default=0)
    rover_final_x = Column(Integer, default=0)
    rover_final_y = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    
    # Relationships
    steps = relationship("MissionStep", back_populates="mission", cascade="all, delete-orphan")
    logs = relationship("MissionLog", back_populates="mission", cascade="all, delete-orphan")
    obstacles = relationship("MissionObstacle", back_populates="mission", cascade="all, delete-orphan")
    report = relationship("MissionReport", back_populates="mission", uselist=False, cascade="all, delete-orphan")


class MissionStep(Base):
    """Mission steps table - stores individual mission steps"""
    __tablename__ = "mission_steps"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    mission_id = Column(String(255), ForeignKey("missions.mission_id", ondelete="CASCADE"), nullable=False, index=True)
    step_number = Column(Integer, nullable=False)
    action = Column(String(100), nullable=False)  # move, explore, scan, collect, return
    target_x = Column(Integer, nullable=True)
    target_y = Column(Integer, nullable=True)
    description = Column(Text, nullable=True)
    completed = Column(Boolean, default=False, nullable=False)
    nasa_image_url = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    mission = relationship("Mission", back_populates="steps")


class MissionLog(Base):
    """Mission activity logs table - stores all agent logs"""
    __tablename__ = "mission_logs"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    mission_id = Column(String(255), ForeignKey("missions.mission_id", ondelete="CASCADE"), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    agent_type = Column(String(50), nullable=False)  # planner, rover, safety, reporter, supervisor, system
    message = Column(Text, nullable=False)
    level = Column(String(20), nullable=False, default="info")  # info, warning, error, success
    data = Column(JSON, nullable=True)  # Additional structured data
    
    # Relationship
    mission = relationship("Mission", back_populates="logs")


class MissionObstacle(Base):
    """Mission obstacles table - stores detected obstacles"""
    __tablename__ = "mission_obstacles"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    mission_id = Column(String(255), ForeignKey("missions.mission_id", ondelete="CASCADE"), nullable=False, index=True)
    x = Column(Integer, nullable=False)
    y = Column(Integer, nullable=False)
    detected_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    mission = relationship("Mission", back_populates="obstacles")


class MissionReport(Base):
    """Mission reports table - stores final mission reports"""
    __tablename__ = "mission_reports"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    mission_id = Column(String(255), ForeignKey("missions.mission_id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    summary = Column(Text, nullable=True)
    outcome = Column(Text, nullable=True)
    completed_steps = Column(Integer, default=0, nullable=False)
    total_steps = Column(Integer, default=0, nullable=False)
    mission_status = Column(String(50), nullable=False)  # complete, aborted, error
    rover_final_x = Column(Integer, nullable=True)
    rover_final_y = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    collected_data = Column(JSON, nullable=True)  # Store collected samples and findings
    mission_photos = Column(JSON, nullable=True)  # Store NASA image URLs and metadata
    apod_data = Column(JSON, nullable=True)  # Store APOD data
    step_details = Column(JSON, nullable=True)  # Store detailed step information
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationship
    mission = relationship("Mission", back_populates="report")

