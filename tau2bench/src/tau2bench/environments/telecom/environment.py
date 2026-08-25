"""Telecom domain environment for tau2bench.

Telecom has two databases — an "agent DB" (customers/lines/plans/bills/devices)
and a "user DB" representing the customer's phone state. The agent only sees
the customer-service tools.
"""

import datetime
import json
import tomllib
import uuid
from collections import defaultdict
from datetime import date, timedelta
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from ac2.runtime import Environment, tool
from tau2bench.dataloader import load_tau2_db
from tau2bench.initial_state import apply_initial_state
from pydantic import BaseModel, Field

FIXED_DATE = date(2025, 2, 25)


def _today() -> date:
    return FIXED_DATE


# ── Agent-side data models ────────────────────────────────────────────────


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str


class Plan(BaseModel):
    plan_id: str
    name: str
    data_limit_gb: float
    price_per_month: float
    data_refueling_price_per_gb: float


class DeviceType(str, Enum):
    PHONE = "phone"
    ROUTER = "router"
    TABLET = "tablet"
    WATCH = "watch"
    OTHER = "other"


class Device(BaseModel):
    device_id: str
    device_type: DeviceType
    model: str
    imei: str | None = None
    is_esim_capable: bool
    activated: bool = False
    activation_date: datetime.datetime | None = None
    last_esim_transfer_date: datetime.datetime | None = None


class LineStatus(str, Enum):
    ACTIVE = "Active"
    SUSPENDED = "Suspended"
    PENDING_ACTIVATION = "Pending Activation"
    CLOSED = "Closed"


class Line(BaseModel):
    line_id: str
    phone_number: str
    status: LineStatus = LineStatus.PENDING_ACTIVATION
    plan_id: str
    device_id: str | None = None
    data_used_gb: float = 0.0
    data_refueling_gb: float = 0.0
    roaming_enabled: bool = False
    contract_end_date: date | None = None
    last_plan_change_date: date | None = None
    last_sim_replacement_date: date | None = None
    suspension_start_date: date | None = None


class LineItem(BaseModel):
    description: str
    amount: float
    date: datetime.date
    item_type: str


class BillStatus(str, Enum):
    DRAFT = "Draft"
    ISSUED = "Issued"
    AWAITING_PAYMENT = "Awaiting Payment"
    PAID = "Paid"
    OVERDUE = "Overdue"
    DISPUTED = "Disputed"


class Bill(BaseModel):
    bill_id: str
    customer_id: str
    period_start: date
    period_end: date
    issue_date: date
    total_due: float
    due_date: date
    line_items: list[LineItem] = Field(default_factory=list)
    status: BillStatus = BillStatus.DRAFT


class AccountStatus(str, Enum):
    ACTIVE = "Active"
    SUSPENDED = "Suspended"
    PENDING_VERIFICATION = "Pending Verification"
    CLOSED = "Closed"


class PaymentMethodType(str, Enum):
    CREDIT_CARD = "Credit Card"
    DEBIT_CARD = "Debit Card"
    PAYPAL = "PayPal"


class PaymentMethod(BaseModel):
    method_type: PaymentMethodType
    account_number_last_4: str
    expiration_date: str


DEFAULT_START_DATE = datetime.date(2025, 1, 1)


class Customer(BaseModel):
    customer_id: str
    full_name: str
    date_of_birth: str
    email: str
    phone_number: str
    address: Address
    account_status: AccountStatus = AccountStatus.PENDING_VERIFICATION
    payment_methods: list[PaymentMethod] = Field(default_factory=list)
    line_ids: list[str] = Field(default_factory=list)
    bill_ids: list[str] = Field(default_factory=list)
    created_at: datetime.datetime = Field(default=DEFAULT_START_DATE)
    last_extension_date: date | None = None
    goodwill_credit_used_this_year: float = 0.0


class TelecomDB(BaseModel):
    plans: list[Plan] = Field(default_factory=list)
    customers: list[Customer] = Field(default_factory=list)
    lines: list[Line] = Field(default_factory=list)
    bills: list[Bill] = Field(default_factory=list)
    devices: list[Device] = Field(default_factory=list)


# ── User-side (phone) data models ─────────────────────────────────────────


class UserSimStatus(str, Enum):
    ACTIVE = "active"
    MISSING = "missing"
    LOCKED_PIN = "locked_pin"
    LOCKED_PUK = "locked_puk"


class UserNetworkTechnology(str, Enum):
    NONE = "none"
    TWO_G = "2G"
    THREE_G = "3G"
    FOUR_G = "4G"
    FIVE_G = "5G"


class UserNetworkModePreference(str, Enum):
    FOUR_G_5G_PREFERRED = "4g_5g_preferred"
    FOUR_G_ONLY = "4g_only"
    THREE_G_ONLY = "3g_only"
    TWO_G_ONLY = "2g_only"


class UserSignalStrength(str, Enum):
    NONE = "none"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class UserNetworkStatus(str, Enum):
    CONNECTED = "connected"
    SEARCHING = "searching"
    NO_SERVICE = "no_service"
    EMERGENCY_ONLY = "emergency_only"


def normalize_network_status(expected_status: str) -> UserNetworkStatus | None:
    """Normalize task/human-facing status strings into the telecom enum.

    Upstream tau2 tasks sometimes assert service status with user-facing
    labels like ``"Active"`` or ``"Connected"`` instead of the internal enum
    values we store on the simulated phone. Without this mapping, a strict
    ``UserNetworkStatus(expected_status)`` call raises ``ValueError`` on those
    aliases and the env assertion silently fails.
    """
    normalized = expected_status.strip().lower().replace(" ", "_").replace("-", "_")
    alias_map = {
        "active": UserNetworkStatus.CONNECTED,
        "connected": UserNetworkStatus.CONNECTED,
        "online": UserNetworkStatus.CONNECTED,
        "searching": UserNetworkStatus.SEARCHING,
        "no_service": UserNetworkStatus.NO_SERVICE,
        "no_signal": UserNetworkStatus.NO_SERVICE,
        "offline": UserNetworkStatus.NO_SERVICE,
        "emergency_only": UserNetworkStatus.EMERGENCY_ONLY,
        "emergency": UserNetworkStatus.EMERGENCY_ONLY,
    }
    if normalized in alias_map:
        return alias_map[normalized]
    try:
        return UserNetworkStatus(normalized)
    except ValueError:
        return None


class UserPerformanceLevel(str, Enum):
    UNKNOWN = "unknown"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class UserAPNNames(str, Enum):
    INTERNET = "internet"
    BROKEN = "broken"


class UserAPNSettings(BaseModel):
    apn_name: UserAPNNames = Field(default=UserAPNNames.INTERNET)
    reset_at_reboot: bool = False
    mmsc_url: str | None = "http://mms.carrier.com/mms/wapenc"


class UserVpnDetails(BaseModel):
    server_address: str | None = None
    protocol: str | None = None
    server_performance: UserPerformanceLevel = UserPerformanceLevel.UNKNOWN


class UserAppPermissions(BaseModel):
    sms: bool = False
    storage: bool = False
    phone: bool = False
    network: bool = False


class UserAppStatus(BaseModel):
    app_name: str
    permissions: UserAppPermissions = Field(default_factory=UserAppPermissions)


class MockPhoneAttributes(BaseModel):
    airplane_mode: bool = False
    sim_card_missing: bool = False
    sim_card_status: UserSimStatus = UserSimStatus.ACTIVE
    network_connection_status: UserNetworkStatus = UserNetworkStatus.CONNECTED
    network_signal_strength: UserSignalStrength = UserSignalStrength.EXCELLENT
    network_technology_connected: UserNetworkTechnology = UserNetworkTechnology.FOUR_G
    network_mode_preference: UserNetworkModePreference = UserNetworkModePreference.FOUR_G_5G_PREFERRED
    data_enabled: bool = True
    roaming_enabled: bool = False
    data_saver_mode: bool = False
    wifi_enabled: bool = False
    wifi_connected: bool = False
    wifi_ssid: str | None = None
    wifi_signal_strength: UserSignalStrength = UserSignalStrength.NONE
    wifi_calling_enabled: bool = False
    wifi_calling_mms_over_wifi: bool = False
    vpn_enabled_setting: bool = False
    vpn_connected: bool = False
    vpn_details: UserVpnDetails | None = None
    active_apn_settings: UserAPNSettings = Field(default_factory=UserAPNSettings)
    app_statuses: dict[str, UserAppStatus] = Field(default_factory=dict)
    battery_level: int = 80


def _default_signal_strength() -> dict[str, UserSignalStrength]:
    return {
        "5G": UserSignalStrength.EXCELLENT,
        "4G": UserSignalStrength.EXCELLENT,
        "3G": UserSignalStrength.GOOD,
        "2G": UserSignalStrength.FAIR,
    }


class UserSurroundings(BaseModel):
    name: str | None = None
    phone_number: str | None = None
    is_abroad: bool = False
    roaming_allowed: bool = True
    mobile_data_usage_exceeded: bool = False
    line_active: bool = True
    signal_strength: dict[str, UserSignalStrength] = Field(default_factory=_default_signal_strength)
    payment_request: dict[str, Any] | None = None


class TelecomUserDB(BaseModel):
    device: MockPhoneAttributes = Field(default_factory=MockPhoneAttributes)
    surroundings: UserSurroundings = Field(default_factory=UserSurroundings)


# ── Helpers ──────────────────────────────────────────────────────────────


def _model_to_json(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), default=str, indent=2)
    if isinstance(obj, list):
        return json.dumps(
            [item.model_dump() if hasattr(item, "model_dump") else item for item in obj],
            default=str,
            indent=2,
        )
    if isinstance(obj, dict):
        result = {k: (v.model_dump() if hasattr(v, "model_dump") else v) for k, v in obj.items()}
        return json.dumps(result, default=str, indent=2)
    return json.dumps(obj, default=str, indent=2)


class _IDGenerator:
    def __init__(self) -> None:
        self.id_counter: dict[str, int] = defaultdict(int)

    def get_id(self, id_type: str, id_name: str | None = None) -> str:
        self.id_counter[id_type] += 1
        return f"{id_name or id_type}_{self.id_counter[id_type]}"


# ── Environment ──────────────────────────────────────────────────────────


class TelecomEnvironment(Environment):
    domain = "telecom"

    async def setup(self, env_params: dict) -> None:
        override = env_params.get("db_path")
        db_path = Path(override) if override else load_tau2_db("telecom")
        if db_path.suffix == ".toml":
            with open(db_path, "rb") as f:
                data = tomllib.load(f)
        else:
            with open(db_path) as f:
                data = json.load(f)
        self._db: TelecomDB = TelecomDB.model_validate(data)
        self._user_db: TelecomUserDB = TelecomUserDB()
        self._id_generator = _IDGenerator()
        self.terminated: bool = False
        await apply_initial_state(self, env_params.get("initial_state"))

    async def teardown(self) -> None:
        return None

    def _save_db(self) -> None:
        return None

    def _save_user_db(self) -> None:
        return None

    # ── Lookups ─────────────────────────────────────────────────────────

    def _get_line_by_id(self, line_id: str) -> Line:
        for line in self._db.lines:
            if line.line_id == line_id:
                return line
        raise ValueError(f"Line with ID {line_id} not found")

    def _get_plan_by_id(self, plan_id: str) -> Plan:
        for plan in self._db.plans:
            if plan.plan_id == plan_id:
                return plan
        raise ValueError(f"Plan with ID {plan_id} not found")

    def _get_device_by_id(self, device_id: str) -> Device:
        for device in self._db.devices:
            if device.device_id == device_id:
                return device
        raise ValueError(f"Device with ID {device_id} not found")

    def _get_bill_by_id(self, bill_id: str) -> Bill:
        for bill in self._db.bills:
            if bill.bill_id == bill_id:
                return bill
        raise ValueError(f"Bill with ID {bill_id} not found")

    def _get_customer_by_id_internal(self, customer_id: str) -> Customer:
        for customer in self._db.customers:
            if customer.customer_id == customer_id:
                return customer
        raise ValueError(f"Customer with ID {customer_id} not found")

    def _get_target_line(self, customer_id: str, line_id: str) -> Line:
        customer = self._get_customer_by_id_internal(customer_id)
        if line_id not in customer.line_ids:
            raise ValueError(f"Line {line_id} not found for customer {customer_id}")
        return self._get_line_by_id(line_id)

    def _get_bills_awaiting_payment(self, customer: Customer) -> list[Bill]:
        bills = []
        for bill_id in customer.bill_ids:
            bill = self._get_bill_by_id(bill_id)
            if bill and bill.status == BillStatus.AWAITING_PAYMENT:
                bills.append(bill)
        return bills

    def _apply_one_time_charge(self, customer_id: str, amount: float, description: str) -> None:
        customer = self._get_customer_by_id_internal(customer_id)
        draft_bill = None
        for bill_id in customer.bill_ids:
            bill = self._get_bill_by_id(bill_id)
            if bill and bill.status == BillStatus.DRAFT:
                draft_bill = bill
                break

        if not draft_bill:
            today = _today()
            next_month = today.replace(day=1) + timedelta(days=32)
            next_month = next_month.replace(day=1)
            new_bill_id = f"B{uuid.uuid4().hex[:8]}"
            draft_bill = Bill(
                bill_id=new_bill_id,
                customer_id=customer_id,
                period_start=next_month,
                period_end=next_month.replace(
                    month=next_month.month + 1 if next_month.month < 12 else 1,
                    year=(next_month.year if next_month.month < 12 else next_month.year + 1),
                )
                - timedelta(days=1),
                issue_date=next_month,
                total_due=0,
                due_date=next_month + timedelta(days=14),
                status=BillStatus.DRAFT,
            )
            self._db.bills.append(draft_bill)
            customer.bill_ids.append(new_bill_id)

        line_item = LineItem(
            description=description,
            amount=amount,
            date=_today(),
            item_type="Credit" if amount < 0 else "Charge",
        )
        draft_bill.line_items.append(line_item)
        draft_bill.total_due += amount

    # ── User-side helpers (phone simulation) ────────────────────────────

    def _get_mobile_data_working(self) -> bool:
        device = self._user_db.device
        surroundings = self._user_db.surroundings
        if device.airplane_mode:
            return False
        if device.network_signal_strength == UserSignalStrength.NONE:
            return False
        if device.network_connection_status == UserNetworkStatus.NO_SERVICE:
            return False
        if surroundings.is_abroad:
            if not device.roaming_enabled or not surroundings.roaming_allowed:
                return False
        if not device.data_enabled:
            return False
        if surroundings.mobile_data_usage_exceeded:
            return False
        if not surroundings.line_active:
            return False
        return True

    def _simulate_network_search(self) -> None:
        device = self._user_db.device
        surroundings = self._user_db.surroundings

        if device.sim_card_missing:
            device.network_connection_status = UserNetworkStatus.NO_SERVICE
            device.network_technology_connected = UserNetworkTechnology.NONE
            device.network_signal_strength = UserSignalStrength.NONE
            return
        if device.sim_card_status in (UserSimStatus.LOCKED_PIN, UserSimStatus.LOCKED_PUK):
            device.network_connection_status = UserNetworkStatus.NO_SERVICE
            device.network_technology_connected = UserNetworkTechnology.NONE
            device.network_signal_strength = UserSignalStrength.NONE
            return
        if device.airplane_mode:
            device.network_connection_status = UserNetworkStatus.NO_SERVICE
            device.network_technology_connected = UserNetworkTechnology.NONE
            device.network_signal_strength = UserSignalStrength.NONE
            return
        if device.active_apn_settings.apn_name == UserAPNNames.BROKEN:
            device.network_connection_status = UserNetworkStatus.NO_SERVICE
            device.network_technology_connected = UserNetworkTechnology.NONE
            device.network_signal_strength = UserSignalStrength.NONE
            return
        if not surroundings.line_active:
            device.network_connection_status = UserNetworkStatus.NO_SERVICE
            device.network_technology_connected = UserNetworkTechnology.NONE
            device.network_signal_strength = UserSignalStrength.NONE
            return

        device.network_connection_status = UserNetworkStatus.CONNECTED
        pref = device.network_mode_preference
        if pref == UserNetworkModePreference.FOUR_G_5G_PREFERRED:
            five_g_signal = surroundings.signal_strength.get("5G", UserSignalStrength.NONE)
            if five_g_signal == UserSignalStrength.NONE:
                device.network_technology_connected = UserNetworkTechnology.FOUR_G
                device.network_signal_strength = surroundings.signal_strength.get("4G", UserSignalStrength.EXCELLENT)
            else:
                device.network_technology_connected = UserNetworkTechnology.FIVE_G
                device.network_signal_strength = five_g_signal
        elif pref == UserNetworkModePreference.FOUR_G_ONLY:
            device.network_technology_connected = UserNetworkTechnology.FOUR_G
            device.network_signal_strength = surroundings.signal_strength.get("4G", UserSignalStrength.EXCELLENT)
        elif pref == UserNetworkModePreference.THREE_G_ONLY:
            device.network_technology_connected = UserNetworkTechnology.THREE_G
            device.network_signal_strength = surroundings.signal_strength.get("3G", UserSignalStrength.GOOD)
        elif pref == UserNetworkModePreference.TWO_G_ONLY:
            device.network_technology_connected = UserNetworkTechnology.TWO_G
            device.network_signal_strength = surroundings.signal_strength.get("2G", UserSignalStrength.FAIR)

    def _run_speed_test_internal(self) -> tuple[float | None, str | None]:
        if not self._get_mobile_data_working():
            return None, "No Connection"
        device = self._user_db.device
        base_speed_factor = 1.0
        if device.vpn_connected and device.vpn_details:
            if device.vpn_details.server_performance == UserPerformanceLevel.POOR:
                base_speed_factor *= 0.1
        if device.data_saver_mode:
            base_speed_factor *= 0.2

        tech_speed_map = {
            UserNetworkTechnology.TWO_G: (0.1, 0.4),
            UserNetworkTechnology.THREE_G: (1.0, 5.0),
            UserNetworkTechnology.FOUR_G: (10.0, 100.0),
            UserNetworkTechnology.FIVE_G: (50.0, 500.0),
            UserNetworkTechnology.NONE: (0.0, 0.0),
        }
        min_speed, max_speed = tech_speed_map.get(device.network_technology_connected, (0.0, 0.0))
        signal_factor_map = {
            UserSignalStrength.POOR: 0.2,
            UserSignalStrength.FAIR: 0.5,
            UserSignalStrength.GOOD: 0.8,
            UserSignalStrength.EXCELLENT: 1.0,
            UserSignalStrength.NONE: 0.0,
        }
        signal_factor = signal_factor_map.get(device.network_signal_strength, 0.0)
        simulated_speed = round((min_speed + max_speed) / 2.0 * signal_factor * base_speed_factor, 2)
        if simulated_speed < 1:
            desc = "Very Poor"
        elif simulated_speed < 5:
            desc = "Poor"
        elif simulated_speed < 25:
            desc = "Fair"
        elif simulated_speed < 100:
            desc = "Good"
        else:
            desc = "Excellent"
        return simulated_speed, desc

    def _can_send_mms_internal(self) -> bool:
        device = self._user_db.device
        if not self._get_mobile_data_working():
            return False
        if device.network_technology_connected == UserNetworkTechnology.TWO_G:
            return False
        if device.wifi_calling_enabled and device.wifi_calling_mms_over_wifi:
            return False
        if device.active_apn_settings.mmsc_url is None:
            return False
        msg_app = device.app_statuses.get("messaging")
        if msg_app is None:
            return False
        return msg_app.permissions.storage and msg_app.permissions.sms

    # ── Agent-facing tools ──────────────────────────────────────────────

    @tool("Finds a customer by their primary contact or line phone number.")
    async def get_customer_by_phone(
        self,
        phone_number: Annotated[str, Field(description="The phone number to search for.")],
    ) -> str:
        for customer in self._db.customers:
            if customer.phone_number == phone_number:
                return _model_to_json(customer)
            for line_id in customer.line_ids:
                line = self._get_line_by_id(line_id)
                if line and line.phone_number == phone_number:
                    return _model_to_json(customer)
        raise ValueError(f"Customer with phone number {phone_number} not found")

    @tool("Retrieves a customer directly by their unique ID.")
    async def get_customer_by_id(
        self,
        customer_id: Annotated[str, Field(description="The unique identifier of the customer.")],
    ) -> str:
        return _model_to_json(self._get_customer_by_id_internal(customer_id))

    @tool("Searches for customers by name and DOB.")
    async def get_customer_by_name(
        self,
        full_name: Annotated[str, Field(description="The full name of the customer.")],
        dob: Annotated[str, Field(description="Date of birth for verification, in the format YYYY-MM-DD.")],
    ) -> str:
        matches = [c for c in self._db.customers if c.full_name.lower() == full_name.lower() and c.date_of_birth == dob]
        return _model_to_json(matches)

    @tool("Retrieves the details for a given ID (line, device, bill, customer, or plan).")
    async def get_details_by_id(
        self,
        id: Annotated[str, Field(description="The ID of the object to retrieve.")],
    ) -> str:
        if id.startswith("L"):
            return _model_to_json(self._get_line_by_id(id))
        if id.startswith("D"):
            return _model_to_json(self._get_device_by_id(id))
        if id.startswith("B"):
            return _model_to_json(self._get_bill_by_id(id))
        if id.startswith("C"):
            return _model_to_json(self._get_customer_by_id_internal(id))
        if id.startswith("P"):
            return _model_to_json(self._get_plan_by_id(id))
        raise ValueError(f"Unknown ID format or type: {id}")

    @tool("Suspends a specific line (max 6 months).")
    async def suspend_line(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer who owns the line.")],
        line_id: Annotated[str, Field(description="ID of the line to suspend.")],
        reason: Annotated[str, Field(description="Reason for suspension.")],
    ) -> str:
        target_line = self._get_target_line(customer_id, line_id)
        if target_line.status != LineStatus.ACTIVE:
            raise ValueError("Line must be active to suspend")
        target_line.status = LineStatus.SUSPENDED
        target_line.suspension_start_date = _today()
        self._save_db()
        return _model_to_json(
            {
                "message": "Line suspended successfully. $5/month holding fee will apply.",
                "line": target_line,
            }
        )

    @tool("Resumes a suspended line.")
    async def resume_line(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer who owns the line.")],
        line_id: Annotated[str, Field(description="ID of the line to resume.")],
    ) -> str:
        target_line = self._get_target_line(customer_id, line_id)
        if target_line.status not in (LineStatus.SUSPENDED, LineStatus.PENDING_ACTIVATION):
            raise ValueError("Line must be suspended to resume")
        target_line.status = LineStatus.ACTIVE
        target_line.suspension_start_date = None
        self._save_db()
        return _model_to_json({"message": "Line resumed successfully", "line": target_line})

    @tool("Retrieves a list of the customer's bills, most recent first.")
    async def get_bills_for_customer(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer.")],
        limit: Annotated[int, Field(description="Maximum number of bills to return.")] = 12,
    ) -> str:
        customer = self._get_customer_by_id_internal(customer_id)
        bills = [self._get_bill_by_id(bid) for bid in customer.bill_ids]
        bills.sort(key=lambda b: b.issue_date, reverse=True)
        return _model_to_json(bills[:limit])

    @tool("Sends a payment request to the customer for a specific bill.")
    async def send_payment_request(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer who owns the bill.")],
        bill_id: Annotated[str, Field(description="ID of the bill to send payment request for.")],
    ) -> str:
        customer = self._get_customer_by_id_internal(customer_id)
        bills = self._get_bills_awaiting_payment(customer)
        if len(bills) != 0:
            raise ValueError("A bill is already awaiting payment for this customer")
        if bill_id not in customer.bill_ids:
            raise ValueError(f"Bill {bill_id} not found for customer {customer_id}")
        bill = self._get_bill_by_id(bill_id)
        bill.status = BillStatus.AWAITING_PAYMENT
        self._user_db.surroundings.payment_request = {
            "bill_id": bill_id,
            "amount_due": bill.total_due,
            "paid": False,
        }
        self._save_user_db()
        self._save_db()
        return f"Payment request sent to the customer for bill {bill.bill_id}"

    @tool("Retrieves current billing cycle data usage for a line.")
    async def get_data_usage(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer who owns the line.")],
        line_id: Annotated[str, Field(description="ID of the line to check usage for.")],
    ) -> str:
        target_line = self._get_target_line(customer_id, line_id)
        plan = self._get_plan_by_id(target_line.plan_id)
        today = _today()
        cycle_end_date = date(today.year, today.month + 1 if today.month < 12 else 1, 1) - timedelta(days=1)
        return _model_to_json(
            {
                "line_id": line_id,
                "data_used_gb": target_line.data_used_gb,
                "data_limit_gb": plan.data_limit_gb,
                "data_refueling_gb": target_line.data_refueling_gb,
                "cycle_end_date": cycle_end_date,
            }
        )

    @tool("Enables international roaming on a line.")
    async def enable_roaming(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer who owns the line.")],
        line_id: Annotated[str, Field(description="ID of the line to enable roaming for.")],
    ) -> str:
        target_line = self._get_target_line(customer_id, line_id)
        if target_line.roaming_enabled:
            return "Roaming was already enabled"
        target_line.roaming_enabled = True
        self._save_db()
        return "Roaming enabled successfully"

    @tool("Disables international roaming on a line.")
    async def disable_roaming(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer who owns the line.")],
        line_id: Annotated[str, Field(description="ID of the line to disable roaming for.")],
    ) -> str:
        target_line = self._get_target_line(customer_id, line_id)
        if not target_line.roaming_enabled:
            return "Roaming was already disabled"
        target_line.roaming_enabled = False
        self._save_db()
        return "Roaming disabled successfully"

    @tool("Transfer the user to a human agent.")
    async def transfer_to_human_agents(
        self,
        summary: Annotated[str, Field(description="A summary of the user's issue.")],
    ) -> str:
        self.terminated = True
        return "Transfer successful"

    @tool("Refuels data for a specific line, adding to the customer's bill.")
    async def refuel_data(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer who owns the line.")],
        line_id: Annotated[str, Field(description="ID of the line to refuel data for.")],
        gb_amount: Annotated[float, Field(description="Amount of data to add in gigabytes.")],
    ) -> str:
        target_line = self._get_target_line(customer_id, line_id)
        if gb_amount <= 0:
            raise ValueError("Refuel amount must be positive")
        plan = self._get_plan_by_id(target_line.plan_id)
        charge_amount = gb_amount * plan.data_refueling_price_per_gb
        target_line.data_refueling_gb += gb_amount
        self._apply_one_time_charge(
            customer_id,
            charge_amount,
            f"Data refueling: {gb_amount} GB at ${plan.data_refueling_price_per_gb}/GB",
        )
        self._save_db()
        return _model_to_json(
            {
                "message": f"Successfully added {gb_amount} GB of data for line {line_id} for ${charge_amount:.2f}",
                "new_data_refueling_gb": target_line.data_refueling_gb,
                "charge": charge_amount,
            }
        )

    @tool("Suspends a line for an unpaid bill.")
    async def suspend_line_for_overdue_bill(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer.")],
        line_id: Annotated[str, Field(description="ID of the line to suspend.")],
        new_bill_id: Annotated[str, Field(description="New bill ID to issue as overdue.")],
        contract_ended: Annotated[bool, Field(description="Whether the contract has ended.")],
    ) -> str:
        line = self._get_line_by_id(line_id)
        if line.status != LineStatus.ACTIVE:
            raise ValueError("Line must be active to suspend for unpaid bill")
        plan = self._get_plan_by_id(line.plan_id)
        amount = plan.price_per_month
        if amount <= 0:
            raise ValueError("Amount must be positive for overdue bill")

        customer = self._get_customer_by_id_internal(customer_id)
        for bill_id in customer.bill_ids:
            bill = self._get_bill_by_id(bill_id)
            if bill.status == BillStatus.OVERDUE:
                raise ValueError("Customer already has an overdue bill")

        today = _today()
        first_day_of_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
        last_day_of_last_month = today.replace(day=1) - timedelta(days=1)

        overdue_bill = Bill(
            bill_id=new_bill_id,
            customer_id=customer_id,
            period_start=first_day_of_last_month,
            period_end=last_day_of_last_month,
            issue_date=first_day_of_last_month,
            total_due=0,
            due_date=first_day_of_last_month + timedelta(days=14),
            status=BillStatus.OVERDUE,
        )
        line_item = LineItem(
            description=f"Charge for line {line.line_id}",
            amount=amount,
            date=_today(),
            item_type="Charge" if amount > 0 else "Credit",
        )
        overdue_bill.line_items.append(line_item)
        overdue_bill.total_due += amount
        self._db.bills.append(overdue_bill)
        customer.bill_ids.append(new_bill_id)
        line.status = LineStatus.SUSPENDED
        line.suspension_start_date = _today()
        if contract_ended:
            line.contract_end_date = last_day_of_last_month
        self._save_db()
        return f"Line {line_id} suspended for unpaid bill {new_bill_id}. Contract ended: {contract_ended}"

    @tool("Sets the data usage for a line.")
    async def set_data_usage(
        self,
        customer_id: Annotated[str, Field(description="ID of the customer who owns the line.")],
        line_id: Annotated[str, Field(description="ID of the line.")],
        data_used_gb: Annotated[float, Field(description="Data used in GB to set on the line.")],
    ) -> str:
        target_line = self._get_target_line(customer_id, line_id)
        target_line.data_used_gb = data_used_gb
        self._save_db()
        return f"Data usage set to {data_used_gb} GB for line {line_id}"

    @tool("Installs an app with specific permissions (for setup).")
    async def install_app(
        self,
        app_name: Annotated[str, Field(description="Name of the app to install.")],
        sms: Annotated[bool, Field(description="Grant SMS permission.")] = False,
        storage: Annotated[bool, Field(description="Grant storage permission.")] = False,
        phone: Annotated[bool, Field(description="Grant phone permission.")] = False,
        network: Annotated[bool, Field(description="Grant network permission.")] = False,
    ) -> str:
        permissions = UserAppPermissions(sms=sms, storage=storage, phone=phone, network=network)
        self._user_db.device.app_statuses[app_name] = UserAppStatus(app_name=app_name, permissions=permissions)
        self._save_user_db()
        return (
            f"App '{app_name}' installed with permissions: "
            f"sms={sms}, storage={storage}, phone={phone}, network={network}"
        )

    def toggle_airplane_mode(self) -> str:
        device = self._user_db.device
        current = device.airplane_mode
        device.airplane_mode = not current
        if current:
            device.network_connection_status = UserNetworkStatus.SEARCHING
            device.wifi_connected = False
            device.wifi_ssid = None
            device.wifi_signal_strength = UserSignalStrength.NONE
        else:
            device.wifi_connected = False
            device.wifi_ssid = None
            device.wifi_signal_strength = UserSignalStrength.NONE
            if device.vpn_connected:
                device.vpn_connected = False
                device.vpn_details = None
        self._simulate_network_search()
        self._save_user_db()
        return f"Airplane Mode is now {'ON' if device.airplane_mode else 'OFF'}"

    def toggle_data(self) -> str:
        device = self._user_db.device
        device.data_enabled = not device.data_enabled
        self._simulate_network_search()
        self._save_user_db()
        return f"Mobile Data is now {'ON' if device.data_enabled else 'OFF'}"

    def toggle_roaming(self) -> str:
        device = self._user_db.device
        device.roaming_enabled = not device.roaming_enabled
        self._simulate_network_search()
        self._save_user_db()
        return f"Data Roaming is now {'ON' if device.roaming_enabled else 'OFF'}"

    def check_network_status(self) -> str:
        device = self._user_db.device
        lines = [
            f"Airplane Mode: {'ON' if device.airplane_mode else 'OFF'}",
            f"SIM Card Status: {device.sim_card_status.value}",
            f"Cellular Connection: {device.network_connection_status.value}",
            f"Cellular Signal: {device.network_signal_strength.value}",
            f"Cellular Network Type: {device.network_technology_connected.value}",
            f"Mobile Data Enabled: {'Yes' if device.data_enabled else 'No'}",
            f"Data Roaming Enabled: {'Yes' if device.roaming_enabled else 'No'}",
            f"Wi-Fi Radio: {'ON' if device.wifi_enabled else 'OFF'}",
            f"Wi-Fi Connected: {'Yes' if device.wifi_connected else 'No'}",
        ]
        if device.wifi_connected:
            lines.append(f"Connected Wi-Fi Network: {device.wifi_ssid}")
        return "\n".join(lines)

    def run_speed_test(self) -> str:
        speed, desc = self._run_speed_test_internal()
        if speed is None:
            return f"Speed test failed: {desc or 'Could not determine speed'}"
        return f"Speed Test Result: {speed:.2f} Mbps ({desc})"

    def reboot_device(self) -> str:
        device = self._user_db.device
        lines = []
        if device.active_apn_settings.reset_at_reboot:
            lines.append("Resetting APN settings...")
            device.active_apn_settings = UserAPNSettings()
        lines.append("Restarting network services...")
        device.network_connection_status = UserNetworkStatus.SEARCHING
        self._simulate_network_search()
        self._save_user_db()
        return "\n".join(lines) if lines else "Device rebooted"

    def connect_vpn(self) -> str:
        device = self._user_db.device
        if device.vpn_connected:
            return "VPN already connected"
        device.vpn_connected = True
        device.vpn_details = UserVpnDetails(
            server_address="192.168.1.1",
            protocol="OpenVPN",
            server_performance=UserPerformanceLevel.EXCELLENT,
        )
        self._save_user_db()
        return "VPN connected successfully"

    def disconnect_vpn(self) -> str:
        device = self._user_db.device
        if not device.vpn_connected:
            return "No active VPN connection to disconnect"
        device.vpn_connected = False
        device.vpn_details = None
        self._save_user_db()
        return "VPN disconnected successfully"

    def check_status_bar(self) -> str:
        device = self._user_db.device
        indicators: list[str] = []
        if device.airplane_mode:
            indicators.append("Airplane Mode")
        else:
            signal_map = {
                UserSignalStrength.NONE: "No Signal",
                UserSignalStrength.POOR: "Poor",
                UserSignalStrength.FAIR: "Fair",
                UserSignalStrength.GOOD: "Good",
                UserSignalStrength.EXCELLENT: "Excellent",
            }
            indicators.append(signal_map.get(device.network_signal_strength, "No Signal"))
            if device.network_technology_connected != UserNetworkTechnology.NONE:
                indicators.append(device.network_technology_connected.value)
            if device.data_enabled and device.network_technology_connected != UserNetworkTechnology.NONE:
                indicators.append("Data Enabled")
                if device.data_saver_mode:
                    indicators.append("Data Saver")
            else:
                indicators.append("Data Disabled")
        if device.wifi_enabled and device.wifi_connected:
            indicators.append(f"Connected to {device.wifi_ssid}" if device.wifi_ssid else "Wi-Fi Enabled")
        if device.vpn_connected:
            indicators.append("VPN Connected")
        indicators.append(f"Battery {device.battery_level}%")
        return f"Status Bar: {' | '.join(indicators)}"

    def check_network_mode_preference(self) -> str:
        return f"Network Mode Preference: {self._user_db.device.network_mode_preference.value}"

    def set_network_mode_preference(self, mode: str) -> str:
        try:
            new_mode = UserNetworkModePreference(mode)
        except ValueError:
            valid = ", ".join(m.value for m in UserNetworkModePreference)
            return f"Failed to set network mode: '{mode}' is not valid. Use one of: {valid}"
        self._user_db.device.network_mode_preference = new_mode
        self._simulate_network_search()
        self._save_user_db()
        return f"Preferred Network Mode set to: {new_mode.value}"

    def check_sim_status(self) -> str:
        device = self._user_db.device
        if device.sim_card_missing:
            return "No SIM card detected in the phone."
        status_map = {
            UserSimStatus.ACTIVE: "Your SIM card is active and working.",
            UserSimStatus.MISSING: "No SIM card detected in the phone.",
            UserSimStatus.LOCKED_PIN: "The SIM card is locked with a PIN code.",
            UserSimStatus.LOCKED_PUK: "The SIM card is locked with a PUK code.",
        }
        return status_map.get(device.sim_card_status, f"Unknown SIM status: {device.sim_card_status.value}")

    def reseat_sim_card(self) -> str:
        self._user_db.device.sim_card_missing = False
        self._simulate_network_search()
        self._save_user_db()
        return "SIM card re-seated successfully."

    def toggle_data_saver_mode(self) -> str:
        self._user_db.device.data_saver_mode = not self._user_db.device.data_saver_mode
        self._save_user_db()
        return f"Data Saver Mode is now {'ON' if self._user_db.device.data_saver_mode else 'OFF'}."

    def check_data_restriction_status(self) -> str:
        if self._user_db.device.data_saver_mode:
            return "Data Saver mode is ON (limits data usage)."
        return "Data Saver mode is OFF."

    def check_apn_settings(self) -> str:
        settings = self._user_db.device.active_apn_settings
        return (
            f"Current APN Name: {settings.apn_name.value or 'Not Set'}\n"
            f"MMSC URL (for picture messages): {settings.mmsc_url or 'Not Set'}"
        )

    def set_apn_settings(self, apn_name: str, mmsc_url: str | None = None) -> str:
        settings = self._user_db.device.active_apn_settings
        try:
            settings.apn_name = UserAPNNames(apn_name)
        except ValueError:
            settings.apn_name = UserAPNNames.INTERNET
        if mmsc_url is not None:
            settings.mmsc_url = mmsc_url
        self._simulate_network_search()
        self._save_user_db()
        return "APN settings updated."

    def reset_apn_settings(self) -> str:
        self._user_db.device.active_apn_settings.reset_at_reboot = True
        self._save_user_db()
        return "APN settings will reset at reboot."

    def check_wifi_status(self) -> str:
        device = self._user_db.device
        if not device.wifi_enabled:
            return "Wi-Fi is turned OFF."
        if device.wifi_connected:
            return (
                f"Wi-Fi is ON and connected to '{device.wifi_ssid}'. "
                f"Signal strength: {device.wifi_signal_strength.value}."
            )
        return "Wi-Fi is ON but not connected to any network."

    def toggle_wifi(self) -> str:
        device = self._user_db.device
        if device.airplane_mode:
            return "Cannot change Wi-Fi settings while Airplane Mode is ON."
        device.wifi_enabled = not device.wifi_enabled
        if not device.wifi_enabled:
            device.wifi_connected = False
            device.wifi_ssid = None
            device.wifi_signal_strength = UserSignalStrength.NONE
        self._save_user_db()
        return f"Wi-Fi is now {'ON' if device.wifi_enabled else 'OFF'}."

    def check_wifi_calling_status(self) -> str:
        return f"Wi-Fi Calling is currently turned {'ON' if self._user_db.device.wifi_calling_enabled else 'OFF'}."

    def toggle_wifi_calling(self) -> str:
        device = self._user_db.device
        device.wifi_calling_enabled = not device.wifi_calling_enabled
        self._save_user_db()
        return f"Wi-Fi Calling is now {'ON' if device.wifi_calling_enabled else 'OFF'}."

    def check_vpn_status(self) -> str:
        device = self._user_db.device
        if device.vpn_connected:
            if device.vpn_details:
                return f"VPN is ON and connected. Details: {device.vpn_details.model_dump()}"
            return "VPN is ON and connected (no specific details available)."
        if device.vpn_enabled_setting:
            return "VPN is turned ON in settings, but currently not connected."
        return "VPN is turned OFF."

    def check_installed_apps(self) -> str:
        names = list(self._user_db.device.app_statuses.keys())
        if not names:
            return "No apps installed."
        return f"The following apps are installed on the phone: {', '.join(names)}"

    def check_app_status(self, app_name: str) -> str:
        app_status = self._user_db.device.app_statuses.get(app_name)
        if app_status is None:
            return f"App '{app_name}' not found on this phone."
        lines = [f"Status for App: {app_name}"]
        allowed = [n.replace("_", " ").lower() for n, ok in app_status.permissions.model_dump().items() if ok]
        if not allowed:
            lines.append(" - Permissions: None granted.")
        else:
            lines.append(" - Permissions Granted:")
            lines.extend(f"   - {p}" for p in allowed)
        return "\n".join(lines)

    def check_app_permissions(self, app_name: str) -> str:
        app_status = self._user_db.device.app_statuses.get(app_name)
        if app_status is None:
            return f"App '{app_name}' not found on this phone."
        allowed = [n.replace("_", " ").lower() for n, ok in app_status.permissions.model_dump().items() if ok]
        if not allowed:
            return f"App '{app_name}' currently has no permissions granted."
        return f"App '{app_name}' has permission for: {', '.join(allowed)}."

    def grant_app_permission(self, app_name: str, permission: str) -> str:
        app_status = self._user_db.device.app_statuses.get(app_name)
        permission = permission.lower()
        if app_status is None:
            return f"App '{app_name}' not found. Cannot grant permission."
        available = list(app_status.permissions.model_dump().keys())
        if permission not in available:
            return f"Permission '{permission}' not tracked for app '{app_name}', available: {available}"
        setattr(app_status.permissions, permission, True)
        self._save_user_db()
        return f"Permission '{permission}' granted to app '{app_name}'."

    def can_send_mms(self) -> str:
        return (
            "Your messaging app can send MMS messages."
            if self._can_send_mms_internal()
            else "Your messaging app cannot send MMS messages."
        )

    def check_payment_request(self) -> str:
        pr = self._user_db.surroundings.payment_request
        if pr is None:
            return "No payment request has been made."
        return f"You have a payment request for bill {pr.get('bill_id')} of {pr.get('amount_due')} USD."

    def make_payment(self) -> str:
        pr = self._user_db.surroundings.payment_request
        if pr is None:
            return "You do not have a payment request."
        pr["paid"] = True
        self._save_user_db()
        return f"Payment of {pr.get('amount_due')} USD has been made for bill {pr.get('bill_id')}."

    # ── Setup methods (called during initial-state application) ─────────

    def set_user_info(self, name: str, phone_number: str) -> str:
        self._user_db.surroundings.name = name
        self._user_db.surroundings.phone_number = phone_number
        self._save_user_db()
        return f"User info set: {name}, {phone_number}"

    def set_user_location(self, abroad: bool) -> str:
        self._user_db.surroundings.is_abroad = abroad
        self._save_user_db()
        return f"User location set: {'abroad' if abroad else 'domestic'}"

    def turn_roaming_on(self) -> str:
        device = self._user_db.device
        if not device.roaming_enabled:
            device.roaming_enabled = True
            self._simulate_network_search()
            self._save_user_db()
        return "Data Roaming is now ON"

    def turn_roaming_off(self) -> str:
        device = self._user_db.device
        if device.roaming_enabled:
            device.roaming_enabled = False
            self._simulate_network_search()
            self._save_user_db()
        return "Data Roaming is now OFF"

    def turn_data_on(self) -> str:
        device = self._user_db.device
        if not device.data_enabled:
            device.data_enabled = True
            self._simulate_network_search()
            self._save_user_db()
        return "Mobile Data is now ON"

    def turn_data_off(self) -> str:
        device = self._user_db.device
        if device.data_enabled:
            device.data_enabled = False
            self._simulate_network_search()
            self._save_user_db()
        return "Mobile Data is now OFF"

    def turn_airplane_mode_on(self) -> str:
        device = self._user_db.device
        if not device.airplane_mode:
            device.airplane_mode = True
            device.wifi_connected = False
            device.wifi_ssid = None
            device.wifi_signal_strength = UserSignalStrength.NONE
            if device.vpn_connected:
                device.vpn_connected = False
                device.vpn_details = None
            self._simulate_network_search()
            self._save_user_db()
        return "Airplane Mode is now ON"

    def turn_airplane_mode_off(self) -> str:
        device = self._user_db.device
        if device.airplane_mode:
            device.airplane_mode = False
            device.network_connection_status = UserNetworkStatus.SEARCHING
            self._simulate_network_search()
            self._save_user_db()
        return "Airplane Mode is now OFF"

    def unseat_sim_card(self) -> str:
        self._user_db.device.sim_card_missing = True
        self._simulate_network_search()
        self._save_user_db()
        return "SIM card un-seated successfully."

    def lock_sim_card(self, mode: str) -> str:
        if mode == "pin":
            self._user_db.device.sim_card_status = UserSimStatus.LOCKED_PIN
        elif mode == "puk":
            self._user_db.device.sim_card_status = UserSimStatus.LOCKED_PUK
        else:
            raise ValueError("Mode must be 'pin' or 'puk'")
        self._simulate_network_search()
        self._save_user_db()
        return f"SIM card locked successfully in {mode} mode."

    def turn_data_saver_mode_on(self) -> str:
        self._user_db.device.data_saver_mode = True
        self._save_user_db()
        return "Data Saver Mode is now ON."

    def turn_data_saver_mode_off(self) -> str:
        self._user_db.device.data_saver_mode = False
        self._save_user_db()
        return "Data Saver Mode is now OFF."

    def break_apn_settings(self) -> str:
        self._user_db.device.active_apn_settings.apn_name = UserAPNNames.BROKEN
        self._simulate_network_search()
        self._save_user_db()
        return "APN settings broken."

    def break_apn_mms_setting(self) -> str:
        self._user_db.device.active_apn_settings.mmsc_url = None
        self._save_user_db()
        return "APN MMS setting broken."

    def set_wifi_calling(self, enabled: bool, mms_over_wifi: bool) -> str:
        self._user_db.device.wifi_calling_enabled = enabled
        self._user_db.device.wifi_calling_mms_over_wifi = mms_over_wifi
        self._save_user_db()
        return (
            f"Wi-Fi Calling is now {'ON' if enabled else 'OFF'}. "
            f"MMS over Wi-Fi is now {'ON' if mms_over_wifi else 'OFF'}."
        )

    def break_vpn(self) -> str:
        self._user_db.device.vpn_connected = True
        self._user_db.device.vpn_details = UserVpnDetails(
            server_address="192.168.1.1",
            protocol="OpenVPN",
            server_performance=UserPerformanceLevel.POOR,
        )
        self._save_user_db()
        return "VPN connection broken (poor performance)."

    def remove_app_permission(self, app_name: str, permission: str) -> str:
        app_status = self._user_db.device.app_statuses.get(app_name)
        permission = permission.lower()
        if not app_status:
            return f"App '{app_name}' not found. Cannot remove permission."
        if not hasattr(app_status.permissions, permission):
            return f"Permission '{permission}' not tracked for app '{app_name}'."
        setattr(app_status.permissions, permission, False)
        self._save_user_db()
        return f"Permission '{permission}' removed from app '{app_name}'."

    def set_mobile_data_usage_exceeded(self, exceeded: bool) -> str:
        self._user_db.surroundings.mobile_data_usage_exceeded = exceeded
        self._simulate_network_search()
        self._save_user_db()
        return f"Mobile data usage exceeded: {exceeded}"

    def set_line_active(self, active: bool) -> str:
        self._user_db.surroundings.line_active = active
        self._simulate_network_search()
        self._save_user_db()
        return f"Line active: {active}"

    # ── Assertions (grader-facing) ──────────────────────────────────────

    def assert_airplane_mode_status(self, expected_status: bool) -> bool:
        return self._user_db.device.airplane_mode == expected_status

    def assert_mobile_data_status(self, expected_status: bool) -> bool:
        return self._get_mobile_data_working() == expected_status

    def assert_internet_speed(self, expected_speed: float, expected_desc: str | None = None) -> bool:
        speed, desc = self._run_speed_test_internal()
        if speed is None:
            return False
        if expected_desc is None:
            return speed >= expected_speed
        return speed >= expected_speed and (desc or "").lower() == expected_desc.lower()

    def assert_mobile_roaming_status(self, expected_status: bool) -> bool:
        return self._user_db.device.roaming_enabled == expected_status

    def assert_service_status(self, expected_status: str) -> bool:
        normalized = normalize_network_status(expected_status)
        if normalized is None:
            return False
        return self._user_db.device.network_connection_status == normalized

    def assert_data_refueling_amount(self, customer_id: str, line_id: str, expected_amount: float) -> bool:
        target_line = self._get_target_line(customer_id, line_id)
        return abs(target_line.data_refueling_gb - expected_amount) < 1e-6

    def assert_line_status(self, customer_id: str, line_id: str, expected_status: str) -> bool:
        target_line = self._get_target_line(customer_id, line_id)
        return target_line.status == LineStatus(expected_status)

    def assert_overdue_bill_exists(self, customer_id: str, overdue_bill_id: str) -> bool:
        customer = self._get_customer_by_id_internal(customer_id)
        if overdue_bill_id not in customer.bill_ids:
            return False
        return self._get_bill_by_id(overdue_bill_id).status == BillStatus.OVERDUE

    def assert_no_overdue_bill(self, overdue_bill_id: str) -> bool:
        try:
            return self._get_bill_by_id(overdue_bill_id).status == BillStatus.PAID
        except ValueError:
            return True

    def assert_mobile_data_saver_mode_status(self, expected_status: bool) -> bool:
        return self._user_db.device.data_saver_mode == expected_status

    def assert_internet_not_excellent(self) -> bool:
        _, desc = self._run_speed_test_internal()
        if desc is None:
            return True
        return desc.lower() != "excellent"

    def assert_can_send_mms(self, expected_status: bool) -> bool:
        return self._can_send_mms_internal() == expected_status

    def assert_mobile_data_usage_exceeded(self, expected_status: bool) -> bool:
        return self._user_db.surroundings.mobile_data_usage_exceeded == expected_status

    # ── DB-state accessors (grader-facing) ──────────────────────────────

    def db_state(self) -> dict[str, Any]:
        return self._db.model_dump()

    def user_db_state(self) -> dict[str, Any]:
        return self._user_db.model_dump()
