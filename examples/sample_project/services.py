"""
Business logic services for the sample e-commerce application.

These services sit between the API layer and the domain models.
Medium complexity — some branching, moderate coupling.
"""

from __future__ import annotations

import logging
from typing import Optional

from models import Address, Order, OrderStatus, PaymentStatus, Product, User
from utils import format_currency, generate_id, validate_email

logger = logging.getLogger(__name__)


class InventoryService:
    """Manages product stock and availability."""

    def __init__(self) -> None:
        self._catalog: dict[str, Product] = {}

    def add_product(self, product: Product) -> None:
        """Add or update a product in the catalog."""
        self._catalog[product.product_id] = product
        logger.info(f"Product {product.product_id} added to catalog.")

    def get_product(self, product_id: str) -> Optional[Product]:
        """Retrieve a product by ID."""
        return self._catalog.get(product_id)

    def restock(self, product_id: str, quantity: int) -> None:
        """Add stock to an existing product."""
        product = self.get_product(product_id)
        if product is None:
            raise KeyError(f"Product {product_id!r} not found.")
        product.stock += quantity

    def reserve_stock(self, product_id: str, quantity: int) -> bool:
        """Attempt to reserve stock for an order. Returns False if insufficient."""
        product = self.get_product(product_id)
        if product is None or product.stock < quantity:
            return False
        product.stock -= quantity
        return True

    def search(self, query: str) -> list[Product]:
        """Full-text search over product names and descriptions."""
        q = query.lower()
        return [
            p for p in self._catalog.values()
            if q in p.name.lower() or q in p.description.lower()
        ]


class UserService:
    """Manages user accounts."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}

    def register(self, email: str, name: str) -> User:
        """Create a new user account."""
        if not validate_email(email):
            raise ValueError(f"Invalid email address: {email!r}")
        user_id = generate_id("usr_")
        user = User(user_id=user_id, email=email, name=name)
        self._users[user_id] = user
        logger.info(f"User {user_id} registered.")
        return user

    def get_user(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def deactivate(self, user_id: str) -> None:
        """Soft-delete a user account."""
        user = self.get_user(user_id)
        if user:
            user.is_active = False

    def update_address(self, user_id: str, address: Address) -> None:
        """Update a user's shipping address."""
        user = self.get_user(user_id)
        if user is None:
            raise KeyError(f"User {user_id!r} not found.")
        user.address = address


class OrderService:
    """
    Orchestrates order creation, modification, and payment.

    This service has higher coupling than InventoryService or UserService —
    it depends on both. Higher coupling → higher structural risk.
    """

    def __init__(
        self,
        inventory: InventoryService,
        user_service: UserService,
    ) -> None:
        self._inventory = inventory
        self._user_service = user_service
        self._orders: dict[str, Order] = {}

    def create_order(self, user_id: str) -> Order:
        """Create an empty order for a user."""
        user = self._user_service.get_user(user_id)
        if user is None:
            raise KeyError(f"User {user_id!r} not found.")
        if not user.is_active:
            raise PermissionError("Cannot place orders for inactive users.")
        order_id = generate_id("ord_")
        order = Order(order_id=order_id, user_id=user_id)
        self._orders[order_id] = order
        return order

    def add_item(self, order_id: str, product_id: str, quantity: int) -> None:
        """Add a product to an existing order, reserving stock."""
        order = self._get_or_raise(order_id)
        product = self._inventory.get_product(product_id)
        if product is None:
            raise KeyError(f"Product {product_id!r} not found.")
        if not self._inventory.reserve_stock(product_id, quantity):
            raise ValueError(
                f"Insufficient stock for {product.name}. Requested: {quantity}."
            )
        order.add_item(product, quantity)
        logger.info(
            f"Added {quantity}x {product.name} to order {order_id}. "
            f"Subtotal: {format_currency(order.total)}"
        )

    def checkout(self, order_id: str, shipping_address: Address) -> Order:
        """Finalize order and mark as confirmed."""
        order = self._get_or_raise(order_id)
        if not order.items:
            raise ValueError("Cannot checkout an empty order.")
        order.shipping_address = shipping_address
        order.status = OrderStatus.CONFIRMED
        order.payment_status = PaymentStatus.PAID
        logger.info(f"Order {order_id} confirmed. Total: {format_currency(order.total)}")
        return order

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order and release reserved inventory."""
        order = self._get_or_raise(order_id)
        for item in order.items:
            self._inventory.restock(item.product.product_id, item.quantity)
        order.cancel()

    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)

    def _get_or_raise(self, order_id: str) -> Order:
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"Order {order_id!r} not found.")
        return order
