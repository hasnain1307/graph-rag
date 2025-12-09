from starlette.exceptions import HTTPException


class UniqueKeyViolationException(HTTPException):
    """Exception raised when the unique key constraint is violated"""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class ObjectNotFoundException(HTTPException):
    """Exception raised when the object is not found"""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class DocumentNotFoundException(HTTPException):
    """Exception raised when the object is not found"""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class NonNullableFieldException(HTTPException):
    """Exception raised when the non-nullable field is not provided"""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class InvalidTokenException(HTTPException):
    """Exception raised when the token is invalid"""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class SessionNotFoundException(HTTPException):
    """Exception raised when the session is not found"""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)


class BadRequestException(HTTPException):
    """Exception raised for bad requests to the server."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(status_code=status_code, detail=detail)
