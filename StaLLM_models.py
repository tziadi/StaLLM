# StaLLM_models.py
import json
import os
from datetime import datetime
from typing import Any

from sqlalchemy import create_engine, Column, Integer, Float, String, Text, DateTime, inspect
from sqlalchemy.orm import declarative_base, sessionmaker

DB_PATH = os.getenv("STALLM_DB_PATH", "StaLLM.db")
ENGINE = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
Session = sessionmaker(bind=ENGINE, autocommit=False, autoflush=False)
Base = declarative_base()

class SmellDetectionResult(Base):
    __tablename__ = "smell_detection_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project = Column(String(256))
    language = Column(String(64))
    filename = Column(String(512))
    strategy = Column(String(128))
    precision = Column(Float)
    recall = Column(Float)
    f1 = Column(Float)
    top_k = Column(Integer)
    time_elapsed = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Nouveau
    llm_used = Column(String(256), default="")

    # JSON texte
    issues_detected = Column(Text)          # JSON list
    sonar_detected_smells = Column(Text)    # JSON list

    # Tokens & coût
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    usd_cost = Column(Float, default=0.0)

def _ensure_column(table: str, name: str, ddl: str):
    insp = inspect(ENGINE)
    cols = [c["name"] for c in insp.get_columns(table)] if insp.has_table(table) else []
    if name not in cols:
        with ENGINE.begin() as conn:
            conn.exec_driver_sql(f'ALTER TABLE "{table}" ADD COLUMN {name} {ddl}')

def init_db():
    Base.metadata.create_all(ENGINE)
    # migrations légères
    _ensure_column("smell_detection_results", "llm_used", "VARCHAR(256) DEFAULT ''")
    _ensure_column("smell_detection_results", "issues_detected", "TEXT")
    _ensure_column("smell_detection_results", "sonar_detected_smells", "TEXT")
    _ensure_column("smell_detection_results", "prompt_tokens", "INTEGER DEFAULT 0")
    _ensure_column("smell_detection_results", "completion_tokens", "INTEGER DEFAULT 0")
    _ensure_column("smell_detection_results", "total_tokens", "INTEGER DEFAULT 0")
    _ensure_column("smell_detection_results", "usd_cost", "FLOAT DEFAULT 0.0")

def _to_json_text(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "null"

def save_run_result(
    project: str,
    filename: str,
    strategy: str,
    language: str,
    f1: float,
    precision: float,
    recall: float,
    top_k: int,
    issues_detected: Any,
    time_elapsed: float,
    llm_used: str,
    sonar_detected_smells: Any,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    usd_cost: float = 0.0,
):
    session = Session()
    try:
        row = SmellDetectionResult(
            project=project,
            filename=filename,
            strategy=strategy,
            language=language,
            f1=f1,
            precision=precision,
            recall=recall,
            top_k=top_k,
            time_elapsed=time_elapsed,
            llm_used=llm_used,
            issues_detected=_to_json_text(issues_detected),
            sonar_detected_smells=_to_json_text(sonar_detected_smells),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            usd_cost=usd_cost,
        )
        session.add(row)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
