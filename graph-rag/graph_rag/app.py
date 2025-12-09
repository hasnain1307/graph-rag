from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi_pagination import add_pagination
import uvicorn

# Import routes for different services
from graph_rag.api.products.routes import router as product_router
from graph_rag.logger import logger

# Import exception handlers and database setup
from issm_api_common.api.exceptions import (
    ObjectNotFoundException,
    UniqueKeyViolationException,
)
from issm_common_database_setup.mongo import BeanieDBClient
from graph_rag import injector


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for initializing and cleaning up resources.

    Args:
        app (FastAPI): The FastAPI application

    Yields:
        None
    """
    # Initialize Beanie DB Client
    db_client = injector.get(BeanieDBClient)
    # Initialize Beanie models
    await db_client.init_beanie()
    # Optional: Add any additional startup tasks
    logger.info("Application startup complete")
    yield
    # Optional: Add any cleanup tasks
    logger.info("Application shutdown initiated")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI application
    """
    # Initialize FastAPI application with lifespan
    app = FastAPI(
        title="Dashboard Service API",
        description="Comprehensive API for managing industries, devices, and products",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        # root_path="/digital/dashboard/",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allows all origins
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
    )

    # Add pagination support
    add_pagination(app)

    app.include_router(product_router, prefix="/api/v1/products", tags=["Products"])

    # Custom exception handlers
    @app.exception_handler(ObjectNotFoundException)
    async def object_not_found_exception_handler(request, exc):
        """
        Custom exception handler for ObjectNotFoundException.

        Args:
            request: The incoming request
            exc: The raised exception

        Returns:
            JSONResponse with error details
        """
        return {"status_code": exc.status_code, "detail": exc.detail}

    @app.exception_handler(UniqueKeyViolationException)
    async def unique_key_violation_exception_handler(request, exc):
        """
        Custom exception handler for UniqueKeyViolationException.

        Args:
            request: The incoming request
            exc: The raised exception

        Returns:
            JSONResponse with error details
        """
        return {"status_code": exc.status_code, "detail": exc.detail}

    # Health check endpoint
    @app.api_route("/", methods=["GET", "HEAD"])
    async def main():
        return {"Hello": "World"}

    return app


# Create the application
app = create_application()

# Optional: If you want to run with uvicorn directly
if __name__ == "__main__":
    uvicorn.run("graph_rag.app:app", host="0.0.0.0", port=8000, reload=True)
