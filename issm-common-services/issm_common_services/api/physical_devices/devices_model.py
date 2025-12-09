from typing import Optional
import uuid
from beanie import Document
from enum import Enum
from pydantic import Field, validator
from datetime import datetime, timedelta, timezone


def get_pk_time_iso():
    return (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()


def get_default_date():
    return "YYYY-MM-DDTHH:mm:ss"


class Status(str, Enum):
    active = "active"
    disabled = "disabled"


class DeviceType(str, Enum):
    camera = "camera"
    machine = "machine"


class PhysicalDevices(Document):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id: str
    industry_id: str
    device_status: Optional[Status] = Field(default=Status.disabled)
    device_type: Optional[DeviceType] = Field(default=DeviceType.camera)
    user_id: Optional[str] = None
    is_archived: bool = False
    is_deleted: bool = False
    activated_on: str = Field(default_factory=get_pk_time_iso)
    deactivated_on: str = Field(default_factory=get_pk_time_iso)
    total_lifetime: str = Field(default_factory=get_pk_time_iso)
    created_on: str = Field(default_factory=get_pk_time_iso)
    deleted_on: str = Field(default_factory=get_pk_time_iso)
    updated_on: str = Field(default_factory=get_pk_time_iso)

    class Settings:
        collection = "physical_devices"

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
