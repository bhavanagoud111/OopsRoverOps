"""Database repository for mission data operations"""
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional, Dict, Any
from datetime import datetime
from app.database.models import (
    Mission, MissionStep, MissionLog, MissionObstacle, MissionReport
)
from app.models.schemas import MissionState, MissionStep as MissionStepSchema, MissionLog as MissionLogSchema, MissionStatus


class MissionRepository:
    """Repository for mission database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_mission(self, mission_id: str, goal: str, status: MissionStatus = MissionStatus.PENDING) -> Mission:
        """Create a new mission record"""
        mission = Mission(
            mission_id=mission_id,
            goal=goal,
            status=status.value,
            current_step=0,
            rover_final_x=0,
            rover_final_y=0,
            start_time=datetime.utcnow()
        )
        self.db.add(mission)
        self.db.commit()
        self.db.refresh(mission)
        return mission
    
    def get_mission(self, mission_id: str) -> Optional[Mission]:
        """Get mission by ID"""
        return self.db.query(Mission).filter(Mission.mission_id == mission_id).first()
    
    def update_mission_status(self, mission_id: str, status: MissionStatus, rover_position: Optional[Dict[str, int]] = None):
        """Update mission status and optionally rover position"""
        mission = self.get_mission(mission_id)
        if mission:
            mission.status = status.value
            mission.updated_at = datetime.utcnow()
            if status == MissionStatus.COMPLETE or status == MissionStatus.ABORTED:
                mission.end_time = datetime.utcnow()
                if mission.start_time:
                    mission.duration_seconds = (mission.end_time - mission.start_time).total_seconds()
            if rover_position:
                mission.rover_final_x = rover_position.get("x", 0)
                mission.rover_final_y = rover_position.get("y", 0)
            self.db.commit()
    
    def add_step(self, mission_id: str, step: MissionStepSchema) -> MissionStep:
        """Add a mission step"""
        db_step = MissionStep(
            mission_id=mission_id,
            step_number=step.step_number,
            action=step.action,
            target_x=step.target_position.x if step.target_position else None,
            target_y=step.target_position.y if step.target_position else None,
            description=step.description,
            completed=step.completed,
            nasa_image_url=step.nasa_image_url
        )
        self.db.add(db_step)
        self.db.commit()
        self.db.refresh(db_step)
        return db_step
    
    def update_step(self, mission_id: str, step_number: int, completed: bool = None, nasa_image_url: str = None):
        """Update a mission step"""
        step = self.db.query(MissionStep).filter(
            MissionStep.mission_id == mission_id,
            MissionStep.step_number == step_number
        ).first()
        if step:
            if completed is not None:
                step.completed = completed
            if nasa_image_url:
                step.nasa_image_url = nasa_image_url
            self.db.commit()
    
    def add_log(self, mission_id: str, log: MissionLogSchema) -> MissionLog:
        """Add a mission activity log"""
        db_log = MissionLog(
            mission_id=mission_id,
            timestamp=log.timestamp,
            agent_type=log.agent_type.value,
            message=log.message,
            level=log.level,
            data=log.data
        )
        self.db.add(db_log)
        self.db.commit()
        self.db.refresh(db_log)
        return db_log
    
    def get_logs(self, mission_id: str, limit: Optional[int] = None) -> List[MissionLog]:
        """Get mission logs, optionally limited"""
        query = self.db.query(MissionLog).filter(
            MissionLog.mission_id == mission_id
        ).order_by(desc(MissionLog.timestamp))
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def add_obstacle(self, mission_id: str, x: int, y: int) -> MissionObstacle:
        """Add a detected obstacle"""
        # Check if obstacle already exists
        existing = self.db.query(MissionObstacle).filter(
            MissionObstacle.mission_id == mission_id,
            MissionObstacle.x == x,
            MissionObstacle.y == y
        ).first()
        if existing:
            return existing
        
        obstacle = MissionObstacle(
            mission_id=mission_id,
            x=x,
            y=y
        )
        self.db.add(obstacle)
        self.db.commit()
        self.db.refresh(obstacle)
        return obstacle
    
    def get_obstacles(self, mission_id: str) -> List[MissionObstacle]:
        """Get all obstacles for a mission"""
        return self.db.query(MissionObstacle).filter(
            MissionObstacle.mission_id == mission_id
        ).all()
    
    def save_report(self, mission_id: str, report_data: Dict[str, Any]) -> MissionReport:
        """Save mission report to database"""
        # Check if report already exists
        existing = self.db.query(MissionReport).filter(
            MissionReport.mission_id == mission_id
        ).first()
        
        if existing:
            # Update existing report
            existing.summary = report_data.get("summary")
            existing.outcome = report_data.get("outcome")
            existing.completed_steps = report_data.get("completed_steps", 0)
            existing.total_steps = report_data.get("total_steps", 0)
            existing.mission_status = report_data.get("mission_status", "complete")
            existing.rover_final_x = report_data.get("rover_final_position", {}).get("x")
            existing.rover_final_y = report_data.get("rover_final_position", {}).get("y")
            existing.duration_seconds = report_data.get("duration_seconds")
            existing.collected_data = report_data.get("collected_data")
            existing.mission_photos = report_data.get("mission_photos")
            existing.apod_data = report_data.get("apod")
            existing.step_details = report_data.get("step_details")
            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            # Create new report
            report = MissionReport(
                mission_id=mission_id,
                summary=report_data.get("summary"),
                outcome=report_data.get("outcome"),
                completed_steps=report_data.get("completed_steps", 0),
                total_steps=report_data.get("total_steps", 0),
                mission_status=report_data.get("mission_status", "complete"),
                rover_final_x=report_data.get("rover_final_position", {}).get("x"),
                rover_final_y=report_data.get("rover_final_position", {}).get("y"),
                duration_seconds=report_data.get("duration_seconds"),
                collected_data=report_data.get("collected_data"),
                mission_photos=report_data.get("mission_photos"),
                apod_data=report_data.get("apod"),
                step_details=report_data.get("step_details")
            )
            self.db.add(report)
            self.db.commit()
            self.db.refresh(report)
            return report
    
    def get_report(self, mission_id: str) -> Optional[MissionReport]:
        """Get mission report by mission ID"""
        return self.db.query(MissionReport).filter(
            MissionReport.mission_id == mission_id
        ).first()
    
    def get_all_missions(self, limit: Optional[int] = None) -> List[Mission]:
        """Get all missions, optionally limited"""
        query = self.db.query(Mission).order_by(desc(Mission.created_at))
        if limit:
            query = query.limit(limit)
        return query.all()

