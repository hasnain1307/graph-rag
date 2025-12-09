from enum import Enum


class ErrorCodes(str, Enum):
    AUTHENTICATION_REQUIRED = "Authentication required."
    AUTHORIZATION_FAILED = "Authorization failed. User has no access."
    INVALID_TOKEN = "Invalid token."
    TOKEN_EXPIRED = "Token expired."
    INVALID_CREDENTIALS = "Invalid credentials."
    EMAIL_TAKEN = "Email is already taken."
    REFRESH_TOKEN_NOT_VALID = "Refresh token is not valid."
    REFRESH_TOKEN_REQUIRED = "Refresh token is required either in the body or cookie."
    TOKEN_NOT_FOUND = "Token not found"
    SESSION_NOT_FOUND = "Session not found"

    DEVICE_NOT_FOUND = "Device not found"
    DEVICE_ALREADY_EXISTS = "Device already exists."

    PRODUCT_NOT_FOUND = "Product not found"
    PRODUCT_ALREADY_EXISTS = "Product already exists."

    INDUSTRY_NOT_FOUND = "Industry not found"
    INDUSTRY_ALREADY_EXISTS = "Industry already exists."


class ErrorStrings(str, Enum):
    duplicate_key = "duplicate key"
    conflict = "Conflict Error"
