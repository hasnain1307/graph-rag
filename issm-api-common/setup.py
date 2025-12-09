import setuptools

requirements = [
    "alembic>=1.12.1",
    "numpy>=1.26.1",
    "pydantic>=2.4.2",
    "pydantic-settings>=2.0.3",
    "pydantic_core>=2.10.1",
    "python-dateutil>=2.8.2",
    "python-decouple>=3.8",
    "python-dotenv>=1.0.0",
    "SQLAlchemy>=1.4.50",
    "typing_extensions>=4.8.0",
    "uvicorn>=0.23.2",
    "starlette>=0.16.0",
    "ruff",
]

setuptools.setup(
    name="issm-api-common", setup_requires=["pip"], install_requires=requirements
)
