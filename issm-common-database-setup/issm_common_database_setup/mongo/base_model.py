from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class BaseMongoModel(BaseModel):
    @classmethod
    def from_mongo(cls, data: dict):
        if "_id" in data:
            data["id"] = str(data.pop("_id"))
        return cls(**data)


class CreateSchemaType(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_deleted: bool = Field(default=False)
    deleted_on: Optional[datetime] = Field(default=None)
