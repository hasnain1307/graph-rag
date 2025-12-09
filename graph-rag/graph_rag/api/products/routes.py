from fastapi import APIRouter, status, Depends, Query
from typing import Optional
from issm_api_common.api.exceptions import (
    ObjectNotFoundException,
)
from issm_common_services.api.products.products_model import Product
from graph_rag.api.products.dependencies import (
    get_product_service,
    ProductService,
)

router = APIRouter()


@router.post(
    "/",
    tags=["Products"],
    response_model=Product,
    status_code=status.HTTP_201_CREATED,
)
async def create_product(
    product: Product,
    product_service: ProductService = Depends(get_product_service),
):
    """
    Create a new product.
    Raises 409 Conflict error if product already exists.
    """
    new_product = await product_service.create(obj_in=product)
    return new_product


@router.get(
    "/",
    tags=["Products"],
    status_code=status.HTTP_200_OK,
)
async def get_all_products(
    product_service: ProductService = Depends(get_product_service),
    cluster_id: Optional[str] = Query(None),
    product_category: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
) -> list[Product]:
    """
    Retrieve all products with optional filtering by cluster_id, product_category, and user_id.
    Raises 404 if no products are found.
    """
    products = await product_service.get_filtered_products(
        cluster_id=cluster_id, product_category=product_category, user_id=user_id
    )
    return products


@router.get(
    "/{id}",
    tags=["Products"],
    status_code=status.HTTP_200_OK,
)
async def get_product(
    id: str, product_service: ProductService = Depends(get_product_service)
) -> Product:
    """
    Retrieve a single product by its ID.
    Raises 404 if the product is not found.
    """
    product = await product_service.get_product_by_id(product_id=id)
    if not product:
        raise ObjectNotFoundException(
            status_code=404, detail=f"Product with id {id} not found"
        )
    return product


@router.put(
    "/{id}",
    tags=["Products"],
    status_code=status.HTTP_200_OK,
)
async def update_product(
    id: str,
    product: Product,
    product_service: ProductService = Depends(get_product_service),
):
    """
    Update a product by ID.
    Supports soft delete when is_deleted is set to True.
    """
    updated_product = await product_service.update_product(
        product=product, product_id=id
    )
    return updated_product
