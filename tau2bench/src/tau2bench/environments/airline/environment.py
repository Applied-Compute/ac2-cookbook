"""Airline domain environment for tau2bench."""

import json
from pathlib import Path
from typing import Annotated, Any, Literal

from ac2.runtime import Environment, tool
from tau2bench.dataloader import load_tau2_db
from tau2bench.initial_state import apply_initial_state
from pydantic import BaseModel, Field

FlightType = Literal["round_trip", "one_way"]
CabinClass = Literal["business", "economy", "basic_economy"]
Insurance = Literal["yes", "no"]


class AirportCode(BaseModel):
    iata: str
    city: str


class Name(BaseModel):
    first_name: str
    last_name: str


class Payment(BaseModel):
    payment_id: str
    amount: int


class PaymentMethodBase(BaseModel):
    id: str
    source: str
    last_four: str | None = None
    amount: int | None = None


class Certificate(PaymentMethodBase):
    source: Literal["certificate"] = "certificate"
    amount: int


class FlightInfo(BaseModel):
    flight_number: str
    date: str


class Passenger(BaseModel):
    first_name: str
    last_name: str
    dob: str


class ReservationFlight(BaseModel):
    origin: str
    destination: str
    flight_number: str
    date: str
    price: int


class Reservation(BaseModel):
    reservation_id: str
    user_id: str
    origin: str
    destination: str
    flight_type: FlightType
    cabin: CabinClass
    flights: list[ReservationFlight]
    passengers: list[Passenger]
    payment_history: list[Payment]
    created_at: str
    total_baggages: int
    nonfree_baggages: int
    insurance: Insurance
    status: Literal["cancelled"] | None = None


class User(BaseModel):
    user_id: str
    name: Name
    email: str
    address: dict[str, Any]
    dob: str
    membership: str
    payment_methods: dict[str, Any]
    saved_passengers: list[Passenger] = []
    reservations: list[str] = []


class FlightDateStatusAvailable(BaseModel):
    status: Literal["available"] = "available"
    available_seats: dict[str, int]
    prices: dict[str, int]


class FlightDateStatusCancelled(BaseModel):
    status: Literal["cancelled"] = "cancelled"


class Flight(BaseModel):
    flight_number: str
    origin: str
    destination: str
    scheduled_departure_time_est: str
    scheduled_arrival_time_est: str
    dates: dict[str, Any]


class DirectFlight(BaseModel):
    flight_number: str
    origin: str
    destination: str
    status: str
    scheduled_departure_time_est: str
    scheduled_arrival_time_est: str
    available_seats: dict[str, int]
    prices: dict[str, int]
    date: str | None = None


class FlightDB(BaseModel):
    flights: dict[str, Flight]
    users: dict[str, User]
    reservations: dict[str, Reservation]


def _to_json(obj: Any) -> str:
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(), default=str, indent=2)
    if isinstance(obj, list):
        return json.dumps(
            [item.model_dump() if hasattr(item, "model_dump") else item for item in obj],
            default=str,
            indent=2,
        )
    return json.dumps(obj, default=str, indent=2)


def _get_datetime() -> str:
    return "2024-05-15T15:00:00"


def _new_payment_ids() -> list[int]:
    return [3221322, 3221323, 3221324]


class AirlineEnvironment(Environment):
    domain = "airline"

    async def setup(self, env_params: dict) -> None:
        override = env_params.get("db_path")
        db_path = Path(override) if override else load_tau2_db("airline")
        with open(db_path) as f:
            data = json.load(f)
        self._db: FlightDB = FlightDB.model_validate(data)
        self.terminated: bool = False
        await apply_initial_state(self, env_params.get("initial_state"))

    async def teardown(self) -> None:
        return None

    def _save_db(self) -> None:
        return None

    def _get_user(self, user_id: str) -> User:
        if user_id not in self._db.users:
            raise ValueError(f"User {user_id} not found")
        return self._db.users[user_id]

    def _get_reservation(self, reservation_id: str) -> Reservation:
        if reservation_id not in self._db.reservations:
            raise ValueError(f"Reservation {reservation_id} not found")
        return self._db.reservations[reservation_id]

    def _get_flight(self, flight_number: str) -> Flight:
        if flight_number not in self._db.flights:
            raise ValueError(f"Flight {flight_number} not found")
        return self._db.flights[flight_number]

    def _get_flight_instance(self, flight_number: str, date: str) -> dict[str, Any]:
        flight = self._get_flight(flight_number)
        if date not in flight.dates:
            raise ValueError(f"Flight {flight_number} not found on date {date}")
        return flight.dates[date]

    def _new_reservation_id(self) -> str:
        for reservation_id in ["HATHAT", "HATHAU", "HATHAV"]:
            if reservation_id not in self._db.reservations:
                return reservation_id
        raise ValueError("Too many reservations")

    def _search_direct_flights(
        self,
        date: str,
        origin: str | None = None,
        destination: str | None = None,
        leave_after: str | None = None,
    ) -> list[DirectFlight]:
        results = []
        for flight in self._db.flights.values():
            check = (
                (origin is None or flight.origin == origin)
                and (destination is None or flight.destination == destination)
                and (date in flight.dates)
                and (flight.dates[date].get("status") == "available")
                and (leave_after is None or flight.scheduled_departure_time_est >= leave_after)
            )
            if check:
                fd = flight.dates[date]
                results.append(
                    DirectFlight(
                        flight_number=flight.flight_number,
                        origin=flight.origin,
                        destination=flight.destination,
                        status="available",
                        scheduled_departure_time_est=flight.scheduled_departure_time_est,
                        scheduled_arrival_time_est=flight.scheduled_arrival_time_est,
                        available_seats=fd.get("available_seats", {}),
                        prices=fd.get("prices", {}),
                    )
                )
        return results

    def _payment_for_update(self, user: User, payment_id: str, total_price: int) -> Payment | None:
        if payment_id not in user.payment_methods:
            raise ValueError("Payment method not found")
        payment_method = user.payment_methods[payment_id]
        if payment_method.get("source") == "certificate":
            raise ValueError("Certificate cannot be used to update reservation")
        if payment_method.get("source") == "gift_card" and payment_method.get("amount", 0) < total_price:
            raise ValueError("Gift card balance is not enough")
        if payment_method.get("source") == "gift_card":
            payment_method["amount"] -= total_price
        if total_price != 0:
            return Payment(payment_id=payment_id, amount=total_price)
        return None

    @tool("Book a reservation.")
    async def book_reservation(
        self,
        user_id: Annotated[str, Field(description="The ID of the user such as 'sara_doe_496'.")],
        origin: Annotated[str, Field(description="The IATA code for origin city such as 'SFO'.")],
        destination: Annotated[str, Field(description="The IATA code for destination city such as 'JFK'.")],
        flight_type: Annotated[str, Field(description="Type of flight: 'one_way' or 'round_trip'.")],
        cabin: Annotated[str, Field(description="Cabin class: 'basic_economy', 'economy', or 'business'.")],
        flights: Annotated[list[dict], Field(description="Flight objects, each with 'flight_number' and 'date'.")],
        passengers: Annotated[list[dict], Field(description="Passenger objects, each with 'first_name', 'last_name', and 'dob'.")],
        payment_methods: Annotated[list[dict], Field(description="Payment method objects, each with 'payment_id' and 'amount'.")],
        total_baggages: Annotated[int, Field(description="Total number of baggage items.")],
        nonfree_baggages: Annotated[int, Field(description="Number of non-free baggage items.")],
        insurance: Annotated[str, Field(description="Whether reservation has insurance: 'yes' or 'no'.")],
    ) -> str:
        flights_list = flights
        passengers_list = passengers
        payment_methods_list = payment_methods

        user = self._get_user(user_id)
        reservation_id = self._new_reservation_id()

        reservation = Reservation(
            reservation_id=reservation_id,
            user_id=user_id,
            origin=origin,
            destination=destination,
            flight_type=flight_type,
            cabin=cabin,
            flights=[],
            passengers=[Passenger(**p) for p in passengers_list],
            payment_history=[Payment(**p) for p in payment_methods_list],
            created_at=_get_datetime(),
            total_baggages=total_baggages,
            nonfree_baggages=nonfree_baggages,
            insurance=insurance,
        )

        total_price = 0
        all_flights_date_data = []

        for flight_info in flights_list:
            flight_number = flight_info["flight_number"]
            flight = self._get_flight(flight_number)
            flight_date_data = self._get_flight_instance(flight_number, flight_info["date"])

            if flight_date_data.get("status") != "available":
                raise ValueError(f"Flight {flight_number} not available on date {flight_info['date']}")
            if flight_date_data["available_seats"].get(cabin, 0) < len(passengers_list):
                raise ValueError(f"Not enough seats on flight {flight_number}")

            price = flight_date_data["prices"][cabin]
            reservation.flights.append(
                ReservationFlight(
                    origin=flight.origin,
                    destination=flight.destination,
                    flight_number=flight_number,
                    date=flight_info["date"],
                    price=price,
                )
            )
            all_flights_date_data.append(flight_date_data)
            total_price += price * len(passengers_list)

        if insurance == "yes":
            total_price += 30 * len(passengers_list)
        total_price += 50 * nonfree_baggages

        for pm in payment_methods_list:
            payment_id = pm["payment_id"]
            amount = pm["amount"]
            if payment_id not in user.payment_methods:
                raise ValueError(f"Payment method {payment_id} not found")
            user_pm = user.payment_methods[payment_id]
            if user_pm.get("source") in {"gift_card", "certificate"}:
                if user_pm.get("amount", 0) < amount:
                    raise ValueError(f"Not enough balance in payment method {payment_id}")

        total_payment = sum(pm["amount"] for pm in payment_methods_list)
        if total_payment != total_price:
            raise ValueError(f"Payment amount does not add up, total price is {total_price}, but paid {total_payment}")

        for pm in payment_methods_list:
            payment_id = pm["payment_id"]
            amount = pm["amount"]
            user_pm = user.payment_methods[payment_id]
            if user_pm.get("source") == "gift_card":
                user_pm["amount"] -= amount
            elif user_pm.get("source") == "certificate":
                del user.payment_methods[payment_id]

        for flight_date_data in all_flights_date_data:
            flight_date_data["available_seats"][cabin] -= len(passengers_list)

        self._db.reservations[reservation_id] = reservation
        self._db.users[user_id].reservations.append(reservation_id)
        self._save_db()
        return _to_json(reservation)

    @tool("Calculate the result of a mathematical expression.")
    async def calculate(
        self,
        expression: Annotated[str, Field(description="The mathematical expression such as '2 + 2'.")],
    ) -> str:
        if not all(char in "0123456789+-*/(). " for char in expression):
            raise ValueError("Invalid characters in expression")
        result = round(float(eval(expression, {"__builtins__": None}, {})), 2)
        return str(result)

    @tool("Cancel the whole reservation.")
    async def cancel_reservation(
        self,
        reservation_id: Annotated[str, Field(description="The reservation ID such as 'ZFA04Y'.")],
    ) -> str:
        reservation = self._get_reservation(reservation_id)
        refunds = [Payment(payment_id=p.payment_id, amount=-p.amount) for p in reservation.payment_history]
        reservation.payment_history.extend(refunds)
        reservation.status = "cancelled"
        self._save_db()
        return _to_json(reservation)

    @tool("Get the details of a reservation.")
    async def get_reservation_details(
        self,
        reservation_id: Annotated[str, Field(description="The reservation ID such as '8JX2WO'.")],
    ) -> str:
        return _to_json(self._get_reservation(reservation_id))

    @tool("Get the details of a user, including their reservations.")
    async def get_user_details(
        self,
        user_id: Annotated[str, Field(description="The user ID such as 'sara_doe_496'.")],
    ) -> str:
        return _to_json(self._get_user(user_id))

    @tool("Returns a list of all available airports.")
    async def list_all_airports(self) -> str:
        airports = [
            AirportCode(iata="SFO", city="San Francisco"),
            AirportCode(iata="JFK", city="New York"),
            AirportCode(iata="LAX", city="Los Angeles"),
            AirportCode(iata="ORD", city="Chicago"),
            AirportCode(iata="DFW", city="Dallas"),
            AirportCode(iata="DEN", city="Denver"),
            AirportCode(iata="SEA", city="Seattle"),
            AirportCode(iata="ATL", city="Atlanta"),
            AirportCode(iata="MIA", city="Miami"),
            AirportCode(iata="BOS", city="Boston"),
            AirportCode(iata="PHX", city="Phoenix"),
            AirportCode(iata="IAH", city="Houston"),
            AirportCode(iata="LAS", city="Las Vegas"),
            AirportCode(iata="MCO", city="Orlando"),
            AirportCode(iata="EWR", city="Newark"),
            AirportCode(iata="CLT", city="Charlotte"),
            AirportCode(iata="MSP", city="Minneapolis"),
            AirportCode(iata="DTW", city="Detroit"),
            AirportCode(iata="PHL", city="Philadelphia"),
            AirportCode(iata="LGA", city="LaGuardia"),
        ]
        return _to_json(airports)

    @tool("Search for direct flights between two cities on a specific date.")
    async def search_direct_flight(
        self,
        origin: Annotated[str, Field(description="Origin city airport code such as 'JFK'.")],
        destination: Annotated[str, Field(description="Destination city airport code such as 'LAX'.")],
        date: Annotated[str, Field(description="Date of flight in format 'YYYY-MM-DD'.")],
    ) -> str:
        results = self._search_direct_flights(date=date, origin=origin, destination=destination)
        return _to_json(results)

    @tool("Search for one-stop flights between two cities on a specific date.")
    async def search_onestop_flight(
        self,
        origin: Annotated[str, Field(description="Origin city airport code such as 'JFK'.")],
        destination: Annotated[str, Field(description="Destination city airport code such as 'LAX'.")],
        date: Annotated[str, Field(description="Date of flight in format 'YYYY-MM-DD'.")],
    ) -> str:
        results = []
        for result1 in self._search_direct_flights(date=date, origin=origin, destination=None):
            result1.date = date
            date2 = f"2024-05-{int(date[-2:]) + 1}" if "+1" in result1.scheduled_arrival_time_est else date
            for result2 in self._search_direct_flights(
                date=date2,
                origin=result1.destination,
                destination=destination,
                leave_after=result1.scheduled_arrival_time_est,
            ):
                result2.date = date2
                results.append([result1, result2])
        return _to_json(results)

    @tool("Send a certificate to a user.")
    async def send_certificate(
        self,
        user_id: Annotated[str, Field(description="The ID of the user such as 'sara_doe_496'.")],
        amount: Annotated[int, Field(description="The amount of the certificate.")],
    ) -> str:
        user = self._get_user(user_id)
        for payment_id in [f"certificate_{i}" for i in _new_payment_ids()]:
            if payment_id not in user.payment_methods:
                certificate = Certificate(id=payment_id, amount=amount)
                user.payment_methods[payment_id] = certificate.model_dump()
                self._save_db()
                return f"Certificate {payment_id} added to user {user_id} with amount {amount}."
        raise ValueError("Too many certificates")

    @tool("Transfer the user to a human agent.")
    async def transfer_to_human_agents(
        self,
        summary: Annotated[str, Field(description="A summary of the user's issue.")],
    ) -> str:
        self.terminated = True
        return "Transfer successful"

    @tool("Update the baggage information of a reservation.")
    async def update_reservation_baggages(
        self,
        reservation_id: Annotated[str, Field(description="The reservation ID such as 'ZFA04Y'.")],
        total_baggages: Annotated[int, Field(description="Updated total number of baggage items.")],
        nonfree_baggages: Annotated[int, Field(description="Updated number of non-free baggage items.")],
        payment_id: Annotated[str, Field(description="The payment id stored in user profile.")],
    ) -> str:
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)
        total_price = 50 * max(0, nonfree_baggages - reservation.nonfree_baggages)

        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation.payment_history.append(payment)

        reservation.total_baggages = total_baggages
        reservation.nonfree_baggages = nonfree_baggages
        self._save_db()
        return _to_json(reservation)

    @tool("Update the flight information of a reservation.")
    async def update_reservation_flights(
        self,
        reservation_id: Annotated[str, Field(description="The reservation ID such as 'ZFA04Y'.")],
        cabin: Annotated[str, Field(description="The cabin class of the reservation.")],
        flights: Annotated[list[dict], Field(description="Flight objects, each with 'flight_number' and 'date'.")],
        payment_id: Annotated[str, Field(description="The payment id stored in user profile.")],
    ) -> str:
        flights_list = flights
        reservation = self._get_reservation(reservation_id)
        user = self._get_user(reservation.user_id)

        total_price = 0
        reservation_flights: list[ReservationFlight] = []

        for flight_info in flights_list:
            matching = next(
                (
                    rf
                    for rf in reservation.flights
                    if rf.flight_number == flight_info["flight_number"]
                    and rf.date == flight_info["date"]
                    and cabin == reservation.cabin
                ),
                None,
            )
            if matching:
                total_price += matching.price * len(reservation.passengers)
                reservation_flights.append(matching)
                continue

            flight = self._get_flight(flight_info["flight_number"])
            flight_date_data = self._get_flight_instance(flight_info["flight_number"], flight_info["date"])

            if flight_date_data.get("status") != "available":
                raise ValueError(f"Flight {flight_info['flight_number']} not available")
            if flight_date_data["available_seats"].get(cabin, 0) < len(reservation.passengers):
                raise ValueError(f"Not enough seats on flight {flight_info['flight_number']}")

            rf = ReservationFlight(
                flight_number=flight_info["flight_number"],
                date=flight_info["date"],
                price=flight_date_data["prices"][cabin],
                origin=flight.origin,
                destination=flight.destination,
            )
            total_price += rf.price * len(reservation.passengers)
            reservation_flights.append(rf)

        total_price -= sum(f.price for f in reservation.flights) * len(reservation.passengers)

        payment = self._payment_for_update(user, payment_id, total_price)
        if payment is not None:
            reservation.payment_history.append(payment)

        reservation.flights = reservation_flights
        reservation.cabin = cabin
        self._save_db()
        return _to_json(reservation)

    @tool("Update the passenger information of a reservation.")
    async def update_reservation_passengers(
        self,
        reservation_id: Annotated[str, Field(description="The reservation ID such as 'ZFA04Y'.")],
        passengers: Annotated[list[dict], Field(description="Passenger objects, each with 'first_name', 'last_name', and 'dob'.")],
    ) -> str:
        passengers_list = passengers
        reservation = self._get_reservation(reservation_id)
        if len(passengers_list) != len(reservation.passengers):
            raise ValueError("Number of passengers does not match")
        reservation.passengers = [Passenger(**p) for p in passengers_list]
        self._save_db()
        return _to_json(reservation)

    @tool("Get the status of a flight.")
    async def get_flight_status(
        self,
        flight_number: Annotated[str, Field(description="The flight number.")],
        date: Annotated[str, Field(description="The date of the flight.")],
    ) -> str:
        flight_instance = self._get_flight_instance(flight_number, date)
        if "status" not in flight_instance:
            raise ValueError(f"Status not found for flight {flight_number} on {date}")
        return flight_instance["status"]

    # ─── grader-facing (not exposed to the agent) ───
    def db_state(self) -> dict[str, Any]:
        """Current DB state for grading (users + reservations only)."""
        return {
            "users": {uid: u.model_dump() for uid, u in self._db.users.items()},
            "reservations": {rid: r.model_dump() for rid, r in self._db.reservations.items()},
        }
