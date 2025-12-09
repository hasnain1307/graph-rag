from pydantic import BaseModel


class ResourceCreatedResponse(BaseModel):
    detail: str
    id: int


class ResourceDeletedResponse(BaseModel):
    detail: str


class ConflictDetail(BaseModel):
    detail: str
