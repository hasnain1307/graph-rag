import setuptools

requirements = [
    "issm-api-common",
    "issm-common-services",
    "beanie==1.29.0",
    "pymongo==4.6.1",
    "motor==3.3.1",
    "fastapi==0.111.0",
    "fastapi-pagination==0.12.14",
    "kafka-python==2.0.2",
    "confluent-kafka==2.3.0",
    "aiokafka==0.10.0",
    "loguru==0.7.2",
    "redis==5.0.3",
    "requests==2.31.0",
    "injector==0.21.0",
    "deepdiff==7.0.1",
    "asyncpg==0.29.0",
    "databases==0.8.0",
    "psycopg2-binary==2.9.9",
    "aiomysql==0.2.0",
    "cryptography==42.0.5",
    "slowapi==0.1.9",
    "zeep==4.2.1",
    "arize-phoenix-evals>=2.0.0",
    "openai>=1.0.0",
    "pandas>=1.0.0",
    "numpy>=1.21.0"
]

setuptools.setup(
    name="llm-as-judge",
    setup_requires=["pip"],
    install_requires=requirements,
    packages=setuptools.find_packages(),  # Add this line
)