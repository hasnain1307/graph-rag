from datetime import datetime, timezone
from typing import List, Optional
from issm_api_common.api.base_service_beanie import BeanieBaseService
from beanie import Document
from typing import TypeVar
from issm_api_common.api.constants import ErrorCodes as ProductErrorCodes
from issm_api_common.api.exceptions import (
    ObjectNotFoundException,
    UniqueKeyViolationException,
)
from issm_common_services.api.products.products_model import Product

ModelType = TypeVar("ModelType", bound=Document)
CreateSchemaType = TypeVar("CreateSchemaType", bound=Document)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=Document)


class ProductService(BeanieBaseService):
    def __init__(self):
        super().__init__(model=Product)

    async def get_filtered_products(self, **filters) -> List[Product]:
        """
        Retrieve a list of products based on provided filters.

        Args:
            **filters: Keyword arguments for filtering products

        Returns:
            List[Product]: List of filtered products
        """
        query = {}
        for key, value in filters.items():
            if value is not None:
                query[key] = value
        return await self.model.find(query).to_list()

    async def get_filtered_product_dictionary(self, **filters) -> Optional[Product]:
        """
        Retrieve a single product based on provided filters.

        Args:
            **filters: Keyword arguments for filtering products

        Returns:
            Optional[Product]: Filtered product or None
        """
        query = {}
        for key, value in filters.items():
            if value is not None:
                query[key] = value

        product = await self.model.find_one(query)
        return product

    async def get_product_by_id(self, product_id: str) -> Product:
        """
        Retrieve a product by its unique identifier.

        Args:
            product_id (str): Unique identifier of the product

        Returns:
            Product: Retrieved product

        Raises:
            ObjectNotFoundException: If product is not found
        """
        product = await self.get(product_id)
        return product

    async def update_product(self, product: Product, product_id: str) -> Product:
        """
        Update a product with additional logic for deletion and archiving.

        Args:
            product (Product): Product to be updated
            product_id (str): Unique identifier of the product

        Returns:
            Product: Updated product

        Raises:
            UniqueKeyViolationException: If product already exists
            ObjectNotFoundException: If product is not found
        """
        try:
            # Automatically archive product if marked as deleted
            if product.is_deleted:
                product.is_archived = True

            # Update timestamp
            product.updated_on = datetime.now(timezone.utc).isoformat()

            # Patch (partially update) the product
            updated_product = await self.patch(id=product_id, obj_in=product)
            return updated_product

        except UniqueKeyViolationException as e:
            # Customize error for product-specific uniqueness violation
            e.detail = ProductErrorCodes.PRODUCT_ALREADY_EXISTS
            raise e

        except ObjectNotFoundException as e:
            # Customize error for product not found
            e.detail = ProductErrorCodes.PRODUCT_NOT_FOUND
            raise e

    async def create_product(self, product: Product) -> Product:
        """
        Create a new product with additional logic.

        Args:
            product (Product): Product to be created

        Returns:
            Product: Created product

        Raises:
            UniqueKeyViolationException: If product already exists
        """
        try:
            # Set creation timestamp
            product.created_on = datetime.now(timezone.utc).isoformat()

            # Create the product
            new_product = await self.create(product)
            return new_product

        except UniqueKeyViolationException as e:
            # Customize error for product-specific uniqueness violation
            e.detail = ProductErrorCodes.PRODUCT_ALREADY_EXISTS
            raise e
