from typing import Optional, Union, Dict
import uuid
from beanie import Document
from pydantic import Field, BaseModel, validator
from enum import Enum
from datetime import datetime, timedelta, timezone


def get_pk_time_iso():
    """
    Get current time in Pakistan Standard Time (UTC+5) in ISO format.

    Returns:
        str: Current timestamp in ISO format
    """
    return (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()


class PackagingType(str, Enum):
    """
    Enumerate different types of product packaging.
    """

    BAG = "bag"
    CARTON = "carton"
    CRATE = "crate"
    BULK = "bulk"
    PALLET = "pallet"
    BOTTLE = "bottle"
    CAN = "can"
    BOX = "box"


class ProductCategory(str, Enum):
    """
    Enumerate broad product categories.
    """

    CONSTRUCTION = "construction"
    AGRICULTURAL = "agricultural"
    FOOD = "food"
    DAIRY = "dairy"
    MANUFACTURING = "manufacturing"
    OTHER = "other"


class ProductPackaging(BaseModel):
    """
    Detailed packaging information for a product.
    """

    type: PackagingType
    quantity_per_package: float = Field(
        gt=0, description="Number of units in a single package"
    )
    package_weight: Optional[float] = Field(
        None, description="Weight of the entire package (in kg)", gt=0
    )
    unit_weight: Optional[float] = Field(
        None, description="Weight of a single unit (in kg)", gt=0
    )
    dimensions: Optional[Dict[str, float]] = Field(
        None, description="Packaging dimensions (length, width, height in meters)"
    )


class ProductSpecifications(BaseModel):
    """
    Flexible product specifications that can be customized for different product types.
    """

    raw_material: Optional[str] = None
    manufacturing_date: Optional[str] = None
    expiry_date: Optional[str] = None
    quality_grade: Optional[str] = None
    additional_details: Optional[Dict[str, Union[str, float, int]]] = None


class Product(Document):
    """
    Comprehensive product schema supporting multiple product types and packaging.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id: str
    camera_id: str
    product_name: str
    product_category: ProductCategory
    user_id: Optional[str] = None
    packaging: ProductPackaging
    specifications: Optional[ProductSpecifications] = None
    is_archived: bool = False
    is_deleted: bool = False
    created_on: str = Field(default_factory=get_pk_time_iso)
    deleted_on: Optional[str] = Field(None)
    updated_on: str = Field(default_factory=get_pk_time_iso)
    supplier: Optional[str] = None
    brand: Optional[str] = None

    class Settings:
        collection = "products"

    class Config:
        arbitrary_types_allowed = True

    @validator("deleted_on", always=True)
    def set_deleted_on(cls, v, values):
        """
        Automatically set deleted_on when is_deleted is True
        """
        if values.get("is_deleted"):
            return get_pk_time_iso()
        return v


# Example usage demonstrations
def create_cement_product():
    """
    Example of creating a cement product with specific packaging and specifications
    """
    return Product(
        cluster_id="cement_cluster_001",
        camera_id="cement_camera_001",
        product_name="Portland Cement",
        product_category=ProductCategory.CONSTRUCTION,
        user_id="user_123",
        packaging=ProductPackaging(
            type=PackagingType.BAG,
            quantity_per_package=1,
            package_weight=50,  # 50 kg bag
            unit_weight=50,
            dimensions={"length": 0.6, "width": 0.4, "height": 0.2},
        ),
        specifications=ProductSpecifications(
            raw_material="Limestone and Clay",
            quality_grade="Type I",
            additional_details={
                "compressive_strength": 42.5,
                "setting_time": 45,  # minutes
            },
        ),
        supplier="Local Cement Works",
        brand="StrongBuild",
    )


def create_egg_product():
    """
    Example of creating an egg product with specific packaging
    """
    return Product(
        cluster_id="eggs_cluster_001",
        camera_id="cement_camera_001",
        product_name="Fresh Eggs",
        product_category=ProductCategory.AGRICULTURAL,
        user_id="user_456",
        packaging=ProductPackaging(
            type=PackagingType.CARTON,
            quantity_per_package=30,  # 30 eggs per carton
            package_weight=2,  # Approximate carton weight
            dimensions={"length": 0.3, "width": 0.2, "height": 0.15},
        ),
        specifications=ProductSpecifications(
            quality_grade="Grade A",
            additional_details={"farm_origin": "Free Range Farm", "egg_size": "Large"},
        ),
        supplier="Green Pastures Farms",
        brand="FarmFresh",
    )
