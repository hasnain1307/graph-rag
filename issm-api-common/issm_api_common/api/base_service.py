from typing import Generic, Optional, Type, TypeVar, Any, List
import sqlalchemy
from pydantic import BaseModel

from issm_api_common.api.constants import ErrorStrings
from issm_api_common.api.exceptions import (
    ObjectNotFoundException,
    UniqueKeyViolationException,
)
from functools import wraps

# Type variables for generic typing
ModelType = TypeVar("ModelType", bound=Any)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


def handle_db_integrity_error(func):
    """
    Decorator to handle database integrity errors consistently.

    Args:
        func (Callable): Method to wrap with error handling

    Returns:
        Callable: Wrapped method with consistent error handling
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except sqlalchemy.exc.IntegrityError as e:
            self.db_object.db.rollback()
            if ErrorStrings.duplicate_key in str(e):
                raise UniqueKeyViolationException(
                    status_code=409, detail=ErrorStrings.conflict
                ) from e
            raise e

    return wrapper


class BaseService(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base service class providing CRUD operations with generic type support.

    Attributes:
        model (Type[ModelType]): SQLAlchemy model class
        db_object: Database connection object
    """

    def __init__(self, model: Type[ModelType], db_object):
        """
        Initialize the base service with a model and database object.

        Args:
            model (Type[ModelType]): SQLAlchemy model class
            db_object: Database connection object
        """
        self.model = model
        self.db_object = db_object

    def _apply_base_filters(self, query):
        """
        Apply base filters to database queries.

        Args:
            query: Base SQLAlchemy query

        Returns:
            Filtered query
        """
        return query.filter(not self.model.is_deleted)

    def get(self, id: int) -> Optional[ModelType]:
        """
        Retrieve a single object by its ID.

        Args:
            id (int): Unique identifier of the object

        Returns:
            Optional[ModelType]: Retrieved object

        Raises:
            ObjectNotFoundException: If object is not found
        """
        obj: Optional[ModelType] = (
            self._apply_base_filters(self.db_object.db.query(self.model))
            .filter(self.model.id == id)
            .first()
        )

        if obj is None:
            raise ObjectNotFoundException(
                status_code=404, detail=f"{self.model.__name__} not found"
            )

        return obj

    def list(self, offset: int = 0, limit: Optional[int] = None) -> List[ModelType]:
        """
        Retrieve a list of objects with optional pagination.

        Args:
            offset (int, optional): Number of records to skip. Defaults to 0.
            limit (Optional[int], optional): Maximum number of records to return.

        Returns:
            List[ModelType]: List of retrieved objects
        """
        query = self._apply_base_filters(self.db_object.db.query(self.model))

        if limit is not None:
            query = query.offset(offset).limit(limit)

        return query.all()

    @handle_db_integrity_error
    def create(self, obj: CreateSchemaType) -> ModelType:
        """
        Create a new database object.

        Args:
            obj (CreateSchemaType): Data for creating the object

        Returns:
            ModelType: Created database object
        """
        db_obj: ModelType = self.model(**obj.dict())
        self.db_object.db.add(db_obj)
        self.db_object.db.commit()
        self.db_object.db.refresh(db_obj)
        return db_obj

    @handle_db_integrity_error
    def update(self, id: int, obj: UpdateSchemaType) -> Optional[ModelType]:
        """
        Update an existing database object.

        Args:
            id (int): Identifier of the object to update
            obj (UpdateSchemaType): Update data

        Returns:
            Optional[ModelType]: Updated database object
        """
        db_obj = self.get(id)

        # Update only non-None values
        update_data = obj.dict(exclude_unset=True)
        for column, value in update_data.items():
            setattr(db_obj, column, value)

        self.db_object.db.commit()
        self.db_object.db.refresh(db_obj)
        return db_obj

    @handle_db_integrity_error
    def delete(self, id: int) -> None:
        """
        Soft delete a database object by marking it as deleted.

        Args:
            id (int): Identifier of the object to delete
        """
        db_obj = self.get(id)
        db_obj.is_deleted = True
        self.db_object.db.add(db_obj)
        self.db_object.db.commit()
