import setuptools

requirements = ["pymongo==4.6.1", "motor==3.3.1", "beanie==1.29.0"]

setuptools.setup(
    name="issm-common-database-setup",
    setup_requires=["pip"],
    install_requires=requirements,
)
