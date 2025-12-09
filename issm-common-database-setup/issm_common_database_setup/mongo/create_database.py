import motor.motor_asyncio
from beanie import init_beanie
from issm_common_services.api.physical_devices.devices_model import PhysicalDevices
from issm_common_services.api.products.products_model import Product
from issm_common_services.api.industry.industry_model import Industry


class BeanieDBClient:
    def __init__(self, conn_str):
        self.db_client = motor.motor_asyncio.AsyncIOMotorClient(conn_str)
        self.db_name = self.db_client.get_database("digital_eye")

    async def init_beanie(self):
        await init_beanie(
            database=self.db_name,
            document_models=[
                Product,
                PhysicalDevices,
                Industry,
            ],
        )
