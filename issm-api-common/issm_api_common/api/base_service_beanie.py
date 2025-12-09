from typing import TypeVar, Generic, Optional, Type, List

import pymongo.errors
from beanie import Document, PydanticObjectId
from pydantic import BaseModel

from issm_api_common.api.exceptions import (
    DocumentNotFoundException,
    UniqueKeyViolationException,
)

# Assuming your BaseMongoModel now extends Document from Beanie
ModelType = TypeVar("ModelType", bound=Document)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BeanieBaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        self.model = model

    async def get(self, id: str) -> Optional[ModelType]:
        obj = await self.model.get(id)
        if obj is None:
            raise DocumentNotFoundException(status_code=404, detail="Not Found")
        return obj

    async def list(self, **filters) -> list[ModelType]:
        query = {}
        for key, value in filters.items():
            if value is not None:
                query[key] = value

        # Apply 'is_deleted' filter only if the attribute exists
        if hasattr(self.model, "is_deleted"):
            query["is_deleted"] = False

        objs = await self.model.find(query, lazy_parse=True).to_list()
        return objs

    async def create(self, obj_in: CreateSchemaType) -> ModelType:
        try:
            obj = self.model(**obj_in.model_dump())
            return await obj.create()
        except pymongo.errors.DuplicateKeyError as e:
            raise UniqueKeyViolationException(
                status_code=409,
                detail=f"{''.join(list(e.details['keyValue'].keys()))} already exists",
            ) from e

    async def create_bulk(self, objects: List[CreateSchemaType]) -> List[ModelType]:
        created_objs = []
        for obj_in in objects:
            try:
                obj = self.model(**obj_in.model_dump())
                await obj.create()
                created_objs.append(obj)
            except pymongo.errors.DuplicateKeyError as e:
                raise UniqueKeyViolationException(
                    status_code=409,
                    detail=f"{''.join(list(e.details['keyValue'].keys()))} already exists",
                ) from e

        return created_objs

    async def patch(self, id: str, obj_in: UpdateSchemaType) -> Optional[ModelType]:
        obj = await self.get(id)
        if obj:
            update_data = obj_in.dict(exclude_unset=True)
            await obj.set(update_data)
            return await self.get(id)
        return None

    async def delete(self, id: str, soft: bool = False) -> None:
        obj = await self.model.find_one(self.model.id == PydanticObjectId(id))
        if obj is None:
            raise DocumentNotFoundException(
                status_code=404, detail="Document not found"
            )

        if soft:
            await obj.set({"is_deleted": True})
        else:
            await obj.delete()
