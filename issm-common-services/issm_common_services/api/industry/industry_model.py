from typing import Optional
import uuid
from beanie import Document
from pydantic import Field, validator
from datetime import datetime, timedelta, timezone


def get_pk_time_iso():
    return (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()


def get_default_date():
    return "YYYY-MM-DDTHH:mm:ss"


class Industry(Document):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id: str
    user_id: str
    industry_category: Optional[str] = None
    industry_name: Optional[str] = None
    devices_count: Optional[int] = None
    is_archived: bool = False
    is_deleted: bool = False
    created_on: str = Field(default_factory=get_pk_time_iso)
    deleted_on: str = Field(default_factory=get_pk_time_iso)
    updated_on: str = Field(default_factory=get_pk_time_iso)

    class Settings:
        collection = "industries"

    class Config:
        arbitrary_types_allowed = True

    @validator("deleted_on", always=True)
    def set_deleted_on(cls, v, values):
        """
        Automatically set deleted_on when is_deleted is True
        """
        if values.get("is_deleted"):
            return get_pk_time_iso()
        return v
