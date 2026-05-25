"""
Data models for the sample e-commerce application.

These are simple dataclasses representing core domain objects.
Well-structured, documented, and low-complexity — expected to be LOW risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class PaymentStatus(Enum):
    UNPAID = "unpaid"
    PAID = "paid"
    REFUNDED = "refunded"


@dataclass
class Address:
    """Shipping or billing address."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "US"


@dataclass
class Product:
    """Represents a purchasable item in the catalog."""
    product_id: str
    name: str
    price: float
    stock: int
    category: str
    description: str = ""

    def is_available(self) -> bool:
        """Check if the product is in stock."""
        return self.stock > 0

    def apply_discount(self, percent: float) -> float:
        """Return discounted price (0–100% range)."""
        if not (0.0 <= percent <= 100.0):
            raise ValueError(f"Discount must be between 0 and 100, got {percent}")
        return self.price * (1 - percent / 100)


@dataclass
class OrderItem:
    """A single line item within an order."""
    product: Product
    quantity: int
    unit_price: float

    @property
    def subtotal(self) -> float:
        """Compute line-item total."""
        return self.unit_price * self.quantity


@dataclass
class Order:
    """An order placed by a user."""
    order_id: str
    user_id: str
    items: list[OrderItem] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    payment_status: PaymentStatus = PaymentStatus.UNPAID
    shipping_address: Optional[Address] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def total(self) -> float:
        """Sum all line-item subtotals."""
        return sum(item.subtotal for item in self.items)

    def add_item(self, product: Product, quantity: int) -> None:
        """Add a product to the order."""
        self.items.append(OrderItem(product, quantity, product.price))

    def cancel(self) -> None:
        """Cancel the order if it hasn't shipped yet."""
        if self.status in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):
            raise RuntimeError("Cannot cancel an order that has already shipped.")
        self.status = OrderStatus.CANCELLED


@dataclass
class User:
    """Application user account."""
    user_id: str
    email: str
    name: str
    address: Optional[Address] = None
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
