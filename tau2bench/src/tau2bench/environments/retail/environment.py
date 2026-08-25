"""Retail domain environment for tau2bench."""

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Union

from ac2.runtime import Environment, tool
from tau2bench.dataloader import load_tau2_db
from tau2bench.initial_state import apply_initial_state
from pydantic import BaseModel, Field


class Variant(BaseModel):
    item_id: str
    options: dict[str, str]
    available: bool
    price: float


class Product(BaseModel):
    name: str
    product_id: str
    variants: dict[str, Variant]


class UserName(BaseModel):
    first_name: str
    last_name: str


class UserAddress(BaseModel):
    address1: str
    address2: str
    city: str
    country: str
    state: str
    zip: str


class PaymentMethodBase(BaseModel):
    source: str
    id: str


class CreditCard(PaymentMethodBase):
    source: Literal["credit_card"] = "credit_card"
    brand: str
    last_four: str


class Paypal(PaymentMethodBase):
    source: Literal["paypal"] = "paypal"


class GiftCard(PaymentMethodBase):
    source: Literal["gift_card"] = "gift_card"
    balance: float


PaymentMethod = Union[CreditCard, GiftCard, Paypal]


class User(BaseModel):
    user_id: str
    name: UserName
    address: UserAddress
    email: str
    payment_methods: dict[str, PaymentMethod]
    orders: list[str]


class OrderFulfillment(BaseModel):
    tracking_id: list[str]
    item_ids: list[str]


class OrderItem(BaseModel):
    name: str
    product_id: str
    item_id: str
    price: float
    options: dict[str, str]


OrderPaymentType = Literal["payment", "refund"]


class OrderPayment(BaseModel):
    transaction_type: OrderPaymentType
    amount: float
    payment_method_id: str


OrderStatus = Literal[
    "processed",
    "pending",
    "pending (item modified)",
    "delivered",
    "cancelled",
    "exchange requested",
    "return requested",
]

CancelReason = Literal["no longer needed", "ordered by mistake"]


class Order(BaseModel):
    order_id: str
    user_id: str
    address: UserAddress
    items: list[OrderItem]
    status: OrderStatus
    fulfillments: list[OrderFulfillment]
    payment_history: list[OrderPayment]
    cancel_reason: CancelReason | None = None
    exchange_items: list[str] | None = None
    exchange_new_items: list[str] | None = None
    exchange_payment_method_id: str | None = None
    exchange_price_difference: float | None = None
    return_items: list[str] | None = None
    return_payment_method_id: str | None = None


class RetailDB(BaseModel):
    products: dict[str, Product]
    users: dict[str, User]
    orders: dict[str, Order]


def _model_to_json(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), default=str, indent=2)
    return json.dumps(obj, default=str, indent=2)


def _is_pending_order(order: Order) -> bool:
    return "pending" in order.status


class RetailEnvironment(Environment):
    domain = "retail"

    async def setup(self, env_params: dict) -> None:
        override = env_params.get("db_path")
        db_path = Path(override) if override else load_tau2_db("retail")
        with open(db_path) as f:
            data = json.load(f)
        self._db: RetailDB = RetailDB.model_validate(data)
        self.terminated: bool = False
        await apply_initial_state(self, env_params.get("initial_state"))

    async def teardown(self) -> None:
        return None

    def _save_db(self) -> None:
        return None

    def _get_order(self, order_id: str) -> Order:
        if order_id not in self._db.orders:
            raise ValueError("Order not found")
        return self._db.orders[order_id]

    def _get_user(self, user_id: str) -> User:
        if user_id not in self._db.users:
            raise ValueError("User not found")
        return self._db.users[user_id]

    def _get_product(self, product_id: str) -> Product:
        if product_id not in self._db.products:
            raise ValueError("Product not found")
        return self._db.products[product_id]

    def _get_variant(self, product_id: str, variant_id: str) -> Variant:
        product = self._get_product(product_id)
        if variant_id not in product.variants:
            raise ValueError("Variant not found")
        return product.variants[variant_id]

    def _get_payment_method(self, user_id: str, payment_method_id: str) -> PaymentMethod:
        user = self._get_user(user_id)
        if payment_method_id not in user.payment_methods:
            raise ValueError("Payment method not found")
        return user.payment_methods[payment_method_id]

    def _get_item(self, item_id: str) -> Variant:
        for product in self._db.products.values():
            if item_id in product.variants:
                return product.variants[item_id]
        raise ValueError("Item not found")

    @tool("Calculate the result of a mathematical expression.")
    async def calculate(
        self,
        expression: Annotated[str, Field(description="The mathematical expression to calculate, such as '2 + 2'.")],
    ) -> str:
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        return str(round(float(eval(expression, {"__builtins__": None}, {})), 2))

    @tool("Cancel a pending order. If the order is already processed or delivered, it cannot be cancelled.")
    async def cancel_pending_order(
        self,
        order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
        reason: Annotated[
            str, Field(description="The reason for cancellation, 'no longer needed' or 'ordered by mistake'.")
        ],
    ) -> str:
        order = self._get_order(order_id)
        if order.status != "pending":
            raise ValueError("Non-pending order cannot be cancelled")
        if reason not in {"no longer needed", "ordered by mistake"}:
            raise ValueError("Invalid reason")

        refunds = []
        for payment in order.payment_history:
            payment_id = payment.payment_method_id
            refund = OrderPayment(
                transaction_type="refund",
                amount=payment.amount,
                payment_method_id=payment_id,
            )
            refunds.append(refund)
            user = self._get_user(order.user_id)
            payment_method = self._get_payment_method(user.user_id, payment_id)
            if isinstance(payment_method, GiftCard):
                payment_method.balance += payment.amount
                payment_method.balance = round(payment_method.balance, 2)

        order.status = "cancelled"
        order.cancel_reason = reason
        order.payment_history.extend(refunds)
        self._save_db()
        return _model_to_json(order)

    @tool("Exchange items in a delivered order to new items of the same product type.")
    async def exchange_delivered_order_items(
        self,
        order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
        item_ids: Annotated[list[str], Field(description="The item ids to be exchanged. There could be duplicate items in the list.")],
        new_item_ids: Annotated[list[str], Field(description="The item ids to be exchanged for.")],
        payment_method_id: Annotated[str, Field(description="The payment method id for the price difference.")],
    ) -> str:
        item_ids_list: list[str] = item_ids
        new_item_ids_list: list[str] = new_item_ids

        order = self._get_order(order_id)
        if order.status != "delivered":
            raise ValueError("Non-delivered order cannot be exchanged")

        all_item_ids = [item.item_id for item in order.items]
        for item_id in item_ids_list:
            if item_ids_list.count(item_id) > all_item_ids.count(item_id):
                raise ValueError(f"Number of {item_id} not found.")

        if len(item_ids_list) != len(new_item_ids_list):
            raise ValueError("The number of items to be exchanged should match.")

        diff_price = 0.0
        for item_id, new_item_id in zip(item_ids_list, new_item_ids_list):
            item = next((it for it in order.items if it.item_id == item_id), None)
            if item is None:
                raise ValueError(f"Item {item_id} not found")
            variant = self._get_variant(item.product_id, new_item_id)
            if not variant.available:
                raise ValueError(f"New item {new_item_id} not found or available")
            diff_price += variant.price - item.price

        diff_price = round(diff_price, 2)
        payment_method = self._get_payment_method(order.user_id, payment_method_id)
        if isinstance(payment_method, GiftCard) and payment_method.balance < diff_price:
            raise ValueError("Insufficient gift card balance to pay for the price difference")

        order.status = "exchange requested"
        order.exchange_items = sorted(item_ids_list)
        order.exchange_new_items = sorted(new_item_ids_list)
        order.exchange_payment_method_id = payment_method_id
        order.exchange_price_difference = diff_price
        self._save_db()
        return _model_to_json(order)

    @tool("Find user id by first name, last name, and zip code.")
    async def find_user_id_by_name_zip(
        self,
        first_name: Annotated[str, Field(description="The first name of the customer.")],
        last_name: Annotated[str, Field(description="The last name of the customer.")],
        zip: Annotated[str, Field(description="The zip code of the customer.")],
    ) -> str:
        for user_id, user in self._db.users.items():
            if (
                user.name.first_name.lower() == first_name.lower()
                and user.name.last_name.lower() == last_name.lower()
                and user.address.zip == zip
            ):
                return user_id
        raise ValueError("User not found")

    @tool("Find user id by email.")
    async def find_user_id_by_email(
        self,
        email: Annotated[str, Field(description="The email of the user.")],
    ) -> str:
        for user_id, user in self._db.users.items():
            if user.email.lower() == email.lower():
                return user_id
        raise ValueError("User not found")

    @tool("Get the status and details of an order.")
    async def get_order_details(
        self,
        order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
    ) -> str:
        return _model_to_json(self._get_order(order_id))

    @tool("Get the inventory details of a product.")
    async def get_product_details(
        self,
        product_id: Annotated[str, Field(description="The product id, such as '6086499569'.")],
    ) -> str:
        return _model_to_json(self._get_product(product_id))

    @tool("Get the inventory details of an item.")
    async def get_item_details(
        self,
        item_id: Annotated[str, Field(description="The item id, such as '6086499569'.")],
    ) -> str:
        return _model_to_json(self._get_item(item_id))

    @tool("Get the details of a user, including their orders.")
    async def get_user_details(
        self,
        user_id: Annotated[str, Field(description="The user id, such as 'sara_doe_496'.")],
    ) -> str:
        return _model_to_json(self._get_user(user_id))

    @tool("List the name and product id of all product types.")
    async def list_all_product_types(self) -> str:
        product_dict = {p.name: p.product_id for p in self._db.products.values()}
        return json.dumps(product_dict, sort_keys=True)

    @tool("Modify the shipping address of a pending order.")
    async def modify_pending_order_address(
        self,
        order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
        address1: Annotated[str, Field(description="The first line of the address.")],
        address2: Annotated[str, Field(description="The second line of the address.")],
        city: Annotated[str, Field(description="The city.")],
        state: Annotated[str, Field(description="The state.")],
        country: Annotated[str, Field(description="The country.")],
        zip: Annotated[str, Field(description="The zip code.")],
    ) -> str:
        order = self._get_order(order_id)
        if not _is_pending_order(order):
            raise ValueError("Non-pending order cannot be modified")
        order.address = UserAddress(
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            country=country,
            zip=zip,
        )
        self._save_db()
        return _model_to_json(order)

    @tool("Modify items in a pending order to new items of the same product type.")
    async def modify_pending_order_items(
        self,
        order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
        item_ids: Annotated[list[str], Field(description="The item ids to be modified. There could be duplicate items in the list.")],
        new_item_ids: Annotated[list[str], Field(description="The item ids to be modified for. There could be duplicate items in the list. Each new item id should match the item id in the same position and be of the same product.")],
        payment_method_id: Annotated[str, Field(description="The payment method id for the price difference.")],
    ) -> str:
        item_ids_list: list[str] = item_ids
        new_item_ids_list: list[str] = new_item_ids

        order = self._get_order(order_id)
        if order.status != "pending":
            raise ValueError("Non-pending order cannot be modified")

        all_item_ids = [item.item_id for item in order.items]
        for item_id in item_ids_list:
            if item_ids_list.count(item_id) > all_item_ids.count(item_id):
                raise ValueError(f"{item_id} not found")

        if len(item_ids_list) != len(new_item_ids_list):
            raise ValueError("The number of items to be exchanged should match")

        diff_price = 0.0
        variant: Variant | None = None
        for item_id, new_item_id in zip(item_ids_list, new_item_ids_list):
            if item_id == new_item_id:
                raise ValueError("The new item id should be different from the old item id")
            item = next((it for it in order.items if it.item_id == item_id), None)
            if item is None:
                raise ValueError(f"Item {item_id} not found")
            variant = self._get_variant(item.product_id, new_item_id)
            if not variant.available:
                raise ValueError(f"New item {new_item_id} not found or available")
            diff_price += variant.price - item.price

        payment_method = self._get_payment_method(order.user_id, payment_method_id)
        if isinstance(payment_method, GiftCard) and payment_method.balance < diff_price:
            raise ValueError("Insufficient gift card balance to pay for the new item")

        order.payment_history.append(
            OrderPayment(
                transaction_type="payment" if diff_price > 0 else "refund",
                amount=abs(diff_price),
                payment_method_id=payment_method_id,
            )
        )
        if isinstance(payment_method, GiftCard):
            payment_method.balance -= diff_price
            payment_method.balance = round(payment_method.balance, 2)

        for item_id, new_item_id in zip(item_ids_list, new_item_ids_list):
            item = next((it for it in order.items if it.item_id == item_id), None)
            if item is None:
                raise ValueError(f"Item {item_id} not found")
            assert variant is not None
            item.item_id = new_item_id
            item.price = variant.price
            item.options = variant.options
        order.status = "pending (item modified)"

        self._save_db()
        return _model_to_json(order)

    @tool("Modify the payment method of a pending order.")
    async def modify_pending_order_payment(
        self,
        order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
        payment_method_id: Annotated[str, Field(description="The payment method id to switch to.")],
    ) -> str:
        order = self._get_order(order_id)
        if not _is_pending_order(order):
            raise ValueError("Non-pending order cannot be modified")
        payment_method = self._get_payment_method(order.user_id, payment_method_id)

        if len(order.payment_history) != 1 or order.payment_history[0].transaction_type != "payment":
            raise ValueError("There should be exactly one payment for a pending order")

        if order.payment_history[0].payment_method_id == payment_method_id:
            raise ValueError("The new payment method should be different from the current one")

        amount = order.payment_history[0].amount

        if isinstance(payment_method, GiftCard) and payment_method.balance < amount:
            raise ValueError("Insufficient gift card balance to pay for the order")

        order.payment_history.extend(
            [
                OrderPayment(
                    transaction_type="payment",
                    amount=amount,
                    payment_method_id=payment_method_id,
                ),
                OrderPayment(
                    transaction_type="refund",
                    amount=amount,
                    payment_method_id=order.payment_history[0].payment_method_id,
                ),
            ]
        )

        if isinstance(payment_method, GiftCard):
            payment_method.balance -= amount
            payment_method.balance = round(payment_method.balance, 2)

        old_payment_method = self._get_payment_method(order.user_id, order.payment_history[0].payment_method_id)
        if isinstance(old_payment_method, GiftCard):
            old_payment_method.balance += amount
            old_payment_method.balance = round(old_payment_method.balance, 2)

        self._save_db()
        return _model_to_json(order)

    @tool("Modify the default address of a user.")
    async def modify_user_address(
        self,
        user_id: Annotated[str, Field(description="The user id, such as 'sara_doe_496'.")],
        address1: Annotated[str, Field(description="The first line of the address.")],
        address2: Annotated[str, Field(description="The second line of the address.")],
        city: Annotated[str, Field(description="The city.")],
        state: Annotated[str, Field(description="The state.")],
        country: Annotated[str, Field(description="The country.")],
        zip: Annotated[str, Field(description="The zip code.")],
    ) -> str:
        user = self._get_user(user_id)
        user.address = UserAddress(
            address1=address1,
            address2=address2,
            city=city,
            state=state,
            country=country,
            zip=zip,
        )
        self._save_db()
        return _model_to_json(user)

    @tool("Return some items of a delivered order.")
    async def return_delivered_order_items(
        self,
        order_id: Annotated[str, Field(description="The order id, such as '#W0000000'.")],
        item_ids: Annotated[list[str], Field(description="The item ids to be returned. There could be duplicate items in the list.")],
        payment_method_id: Annotated[str, Field(description="The payment method id for the refund.")],
    ) -> str:
        item_ids_list: list[str] = item_ids

        order = self._get_order(order_id)
        if order.status != "delivered":
            raise ValueError("Non-delivered order cannot be returned")

        user = self._get_user(order.user_id)
        payment_method = self._get_payment_method(user.user_id, payment_method_id)

        if not isinstance(payment_method, GiftCard) and payment_method_id != order.payment_history[0].payment_method_id:
            raise ValueError("Payment method should be the original payment method")

        all_item_ids = [item.item_id for item in order.items]
        for item_id in item_ids_list:
            if item_ids_list.count(item_id) > all_item_ids.count(item_id):
                raise ValueError("Some item not found")

        order.status = "return requested"
        order.return_items = sorted(item_ids_list)
        order.return_payment_method_id = payment_method_id
        self._save_db()
        return _model_to_json(order)

    @tool("Transfer the user to a human agent.")
    async def transfer_to_human_agents(
        self,
        summary: Annotated[str, Field(description="A summary of the user's issue.")],
    ) -> str:
        self.terminated = True
        return "Transfer successful"

    # ─── grader-facing (not exposed to the agent) ───
    def db_state(self) -> dict[str, Any]:
        """Current DB state for grading (users + orders only)."""
        return {
            "users": {uid: u.model_dump() for uid, u in self._db.users.items()},
            "orders": {oid: o.model_dump() for oid, o in self._db.orders.items()},
        }
