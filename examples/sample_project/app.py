"""
Main application entry point for the sample e-commerce project.

Wires together all services and exposes a simple in-process "API".
This file has the highest fan-in in the sample project — it imports from
every other module — making it a good target for GraphGuard to flag.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from buggy_module import batch_process_files, compute_risk_score, process_order_data
from models import Address, Order, Product, User
from services import InventoryService, OrderService, UserService
from utils import format_currency, generate_id, paginate

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


class Application:
    """
    Top-level application container.

    Initialises all services and provides convenience methods for common
    workflows.  This class has the highest coupling in the codebase because
    it directly references every other module.
    """

    def __init__(self) -> None:
        self.inventory = InventoryService()
        self.users = UserService()
        self.orders = OrderService(self.inventory, self.users)
        logger.info("Application initialised.")

    # ------------------------------------------------------------------
    # User flows
    # ------------------------------------------------------------------

    def register_user(self, email: str, name: str, street: str = "", city: str = "") -> User:
        """Register a new user and optionally set their address."""
        user = self.users.register(email, name)
        if street and city:
            addr = Address(street=street, city=city, state="CA", zip_code="90210")
            self.users.update_address(user.user_id, addr)
        return user

    # ------------------------------------------------------------------
    # Product flows
    # ------------------------------------------------------------------

    def add_product(
        self, name: str, price: float, stock: int, category: str = "general"
    ) -> Product:
        """Create and register a product."""
        product = Product(
            product_id=generate_id("prod_"),
            name=name,
            price=price,
            stock=stock,
            category=category,
        )
        self.inventory.add_product(product)
        return product

    # ------------------------------------------------------------------
    # Order flows
    # ------------------------------------------------------------------

    def place_order(
        self,
        user_id: str,
        items: list[tuple[str, int]],   # [(product_id, quantity), ...]
        shipping: Optional[Address] = None,
    ) -> Order:
        """
        Create and confirm an order in one step.

        Parameters
        ----------
        user_id  : ID of the user placing the order
        items    : list of (product_id, quantity) tuples
        shipping : shipping address (defaults to user's registered address)
        """
        order = self.orders.create_order(user_id)
        for product_id, quantity in items:
            self.orders.add_item(order.order_id, product_id, quantity)

        if shipping is None:
            user = self.users.get_user(user_id)
            shipping = user.address if user else None

        if shipping is None:
            raise ValueError("No shipping address provided and none on file.")

        return self.orders.checkout(order.order_id, shipping)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def order_summary(self, order_id: str) -> dict[str, Any]:
        """Return a human-readable summary of an order."""
        order = self.orders.get_order(order_id)
        if order is None:
            return {"error": f"Order {order_id!r} not found"}
        return {
            "order_id": order.order_id,
            "user_id": order.user_id,
            "status": order.status.value,
            "total": format_currency(order.total),
            "items": [
                {
                    "name": item.product.name,
                    "qty": item.quantity,
                    "subtotal": format_currency(item.subtotal),
                }
                for item in order.items
            ],
        }


# ---------------------------------------------------------------------------
# Demo run
# ---------------------------------------------------------------------------

def run_demo() -> None:
    """Execute a short end-to-end demo that exercises every service."""
    app = Application()

    # Register users
    alice = app.register_user(
        "alice@example.com", "Alice Smith",
        street="123 Main St", city="Los Angeles"
    )
    bob = app.register_user("bob@example.com", "Bob Jones")

    # Add products
    laptop = app.add_product("Laptop Pro 15", 1299.99, 10, "electronics")
    headphones = app.add_product("Noise Cancelling Headphones", 249.99, 25, "electronics")
    book = app.add_product("Clean Code", 34.99, 100, "books")

    logger.info(f"Catalog: {laptop.name}, {headphones.name}, {book.name}")

    # Place an order for Alice
    alice_addr = Address("123 Main St", "Los Angeles", "CA", "90001")
    order = app.place_order(
        alice.user_id,
        [(laptop.product_id, 1), (headphones.product_id, 2)],
        shipping=alice_addr,
    )
    logger.info(f"Order placed: {order.order_id}, total: {format_currency(order.total)}")

    summary = app.order_summary(order.order_id)
    logger.info(f"Summary: {summary}")

    # Demo buggy_module integration
    raw_order_data = {
        "email": alice.email,
        "name": alice.name,
    }
    result = process_order_data(
        raw_order_data,
        user_id=alice.user_id,
        product_ids=[laptop.product_id, book.product_id],
        quantities=[1, 3],
        discount=10.0,
        flags=None,
    )
    logger.info(f"buggy_module result: total={format_currency(result['total'])}")

    # Paginate a product list
    all_products = [laptop, headphones, book]
    page1 = paginate(all_products, page=1, page_size=2)
    logger.info(f"Page 1 products: {[p.name for p in page1]}")


if __name__ == "__main__":
    run_demo()
