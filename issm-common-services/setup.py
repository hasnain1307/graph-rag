import setuptools

requirements = [
    "issm-api-common",
    "httpx",
]

setuptools.setup(
    name="issm-common-services",
    setup_requires=["pip"],
    install_requires=requirements,
)
