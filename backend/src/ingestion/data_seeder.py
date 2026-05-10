"""
Synthetic banking data seeder for DuckDB in-memory execution.

Why synthetic data and not a real dataset:
  Real banking data cannot be shared. Synthetic data lets us demonstrate the full
  pipeline with realistic-looking records that exercise all four use cases
  (understand what changed, compare, decompose, summarize) across all 25 tables.

Actual schema (verified from JSONL):

CustSrv (5 tables):
  Company          — company_id, name, country
  Branch           — branch_id, company_id*, name, region
  CustomerType     — type_id, type_name, description
  Customer         — customer_id, name, type_id*, branch_id*, risk_rating, created_at
  CustomerAddress  — address_id, customer_id*, line1, city, country, postcode

CoreSrv (4 tables):
  Account          — account_id, customer_id*, advisor_id*, account_type, currency,
                     opened_date, wrapper_id*
  AccountBalance   — balance_id, account_id*, balance, as_of_date
  CashMovement     — movement_id, account_id*, amount, direction, timestamp
  TransferRequest  — transfer_id, source_account*, dest_account*, amount, status

WealthSrv (4 tables):
  WrapProvider     — provider_id, name, country
  ProductWrapper   — wrapper_id, provider_id*, wrapper_type, currency
  ProductWrapperType — wrapper_type_id, name
  Advisor          — advisor_id, name, branch_id*, region

TradeSrv (7 tables):
  SecurityType     — security_type_id, type_name
  SecuritySubType  — subtype_id, security_type_id*, name
  Instrument       — instrument_id, name, security_type_id*, currency, subtype_id*
  Order            — order_id, account_id*, instrument_id*, side, quantity
  Trade            — trade_id, order_id*, account_id*, price, quantity, instrument_id*
  AggregateOrder   — agg_order_id, created_at, order_id*
  Batch            — batch_id, run_date, agg_order_id*

AuthSrv (5 tables):
  UserGroup        — group_id, group_name
  Application      — app_id, app_name
  User             — user_id, advisor_id*, name, group_id*
  UserGroupMembership — membership_id, user_id*, group_id*
  Session          — session_id, user_id*, app_id*, login_time

* = foreign key
"""

import duckdb
import random
from datetime import date, timedelta
from typing import Dict, List, Set


ROWS_SMALL = 5
ROWS_MID = 20
ROWS_LARGE = 40
ROWS_CHILD = 60

random.seed(42)

START = date(2023, 1, 1)
END = date(2024, 12, 31)

CUSTOMER_NAMES = [
    "James Smith",
    "Emma Johnson",
    "Oliver Williams",
    "Sophia Brown",
    "William Jones",
    "Amelia Garcia",
    "George Miller",
    "Isabella Davis",
    "Harry Wilson",
    "Mia Anderson",
    "Jack Taylor",
    "Charlotte Thomas",
    "Noah Jackson",
    "Ava White",
    "Liam Harris",
    "Evelyn Martin",
    "Sebastian Thompson",
    "Harper Moore",
    "Alexander Allen",
    "Luna Lewis",
    "Ethan Robinson",
    "Aria Walker",
    "Mason Hall",
    "Chloe Young",
    "Logan King",
    "Penelope Wright",
    "Benjamin Green",
    "Layla Baker",
    "Lucas Hill",
    "Riley Campbell",
    "Daniel Scott",
    "Grace Adams",
    "Henry Nelson",
    "Zoe Carter",
    "Samuel Mitchell",
    "Lily Perez",
    "Owen Roberts",
    "Hannah Turner",
    "Julian Phillips",
    "Natalie Evans",
]

ADVISOR_NAMES = [
    "Catherine Hughes",
    "Michael Foster",
    "Rachel Simmons",
    "David Cooper",
    "Laura Richardson",
    "James Cox",
    "Sarah Ward",
    "Robert Hughes",
    "Emma Patterson",
    "Thomas Griffin",
    "Jennifer Howard",
    "Charles Ross",
    "Amanda Watson",
    "Jonathan Price",
    "Stephanie Butterfield",
    "Kevin Barnes",
    "Natasha Fleming",
    "Andrew Wood",
    "Caroline Shaw",
    "Mark Hudson",
]

COMPANY_NAMES = ["NatWest Group", "Royal Bank of Scotland", "Ulster Bank"]
COUNTRIES = ["United Kingdom", "United Kingdom", "Ireland"]
REGIONS = ["North", "South", "East", "West", "Scotland", "Wales", "London"]
BRANCH_CITIES = [
    "London",
    "Edinburgh",
    "Manchester",
    "Birmingham",
    "Glasgow",
    "Leeds",
    "Bristol",
    "Liverpool",
    "Sheffield",
    "Cardiff",
    "Belfast",
    "Newcastle",
    "Nottingham",
    "Southampton",
    "Leicester",
    "Aberdeen",
    "Dundee",
    "Inverness",
    "Oxford",
    "Cambridge",
]
ACCOUNT_TYPES = ["ISA", "GIA", "PENSION", "SIPP", "JISA"]
CURRENCIES = ["GBP", "GBP", "GBP", "EUR", "USD"]
RISK_RATINGS = ["LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "VERY_HIGH"]
DIRECTIONS = ["CREDIT", "DEBIT"]
TRANSFER_STATUS = ["COMPLETED", "COMPLETED", "PENDING", "FAILED"]
ORDER_SIDES = ["BUY", "SELL"]
WRAP_TYPES = ["ISA", "PENSION", "GIA", "SIPP"]
WRAP_PROVIDERS = ["Hargreaves Lansdown", "AJ Bell", "Fidelity", "Vanguard", "BlackRock"]
SECURITY_TYPES = ["EQUITY", "BOND", "ETF", "FUND"]
SECURITY_SUBTYPES = {
    "EQUITY": ["UK Large Cap", "US Large Cap", "Emerging Markets", "Small Cap"],
    "BOND": ["Government", "Corporate", "High Yield", "Index Linked"],
    "ETF": ["Index ETF", "Active ETF", "Thematic ETF"],
    "FUND": ["Active Fund", "Passive Fund", "Hedge Fund"],
}
INSTRUMENT_NAMES = [
    "Apple Inc",
    "BP plc",
    "HSBC Holdings",
    "Barclays plc",
    "Shell plc",
    "Unilever plc",
    "AstraZeneca plc",
    "GlaxoSmithKline plc",
    "Vodafone Group",
    "BT Group",
    "FTSE 100 ETF",
    "S&P 500 ETF",
    "Global Bond Fund",
    "UK Government Bond",
    "US Treasury Bond",
    "Vanguard LifeStrategy",
    "BlackRock World Index",
    "Fidelity Global Technology",
    "JP Morgan US Growth",
    "Baillie Gifford Positive Change",
]
GROUP_NAMES = ["ADMIN", "ANALYST", "TRADER", "VIEWER", "COMPLIANCE"]
APP_NAMES = [
    "Trading Portal",
    "Customer Portal",
    "Admin Dashboard",
    "Compliance Hub",
    "Reporting Suite",
]
CUST_TYPES = ["RETAIL", "PRIVATE", "CORPORATE", "INSTITUTIONAL"]
CUST_DESCS = [
    "Standard retail customer",
    "High net worth private client",
    "Business or institutional client",
    "Institutional investor",
]


def _date_between(start: date, end: date) -> date:
    return start + timedelta(days=random.randint(0, (end - start).days))


def _ts(start: date = START, end: date = END) -> str:
    d = _date_between(start, end)
    return f"{d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00"


def _amount(lo: float, hi: float) -> float:
    return round(random.uniform(lo, hi), 2)


class BankingDataSeeder:
    """
    Seeds all 25 banking tables into DuckDB in topological FK order.
    Every column name matches banking_tables_typed.jsonl exactly.
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        self.company_ids: List[int] = []
        self.branch_ids: List[int] = []
        self.customer_type_ids: List[int] = []
        self.customer_ids: List[int] = []
        self.wrap_provider_ids: List[int] = []
        self.wrapper_ids: List[int] = []
        self.advisor_ids: List[int] = []
        self.account_ids: List[int] = []
        self.security_type_ids: List[int] = []
        self.subtype_ids: List[int] = []
        self.instrument_ids: List[int] = []
        self.order_ids: List[int] = []
        self.agg_order_ids: List[int] = []
        self.group_ids: List[int] = []
        self.app_ids: List[int] = []
        self.user_ids: List[int] = []

    def seed_all(self) -> Dict[str, int]:
        print("[data_seeder] Creating tables from JSONL schema DDL...")
        self._create_all_tables()
        print("[data_seeder] Seeding in FK topological order...")
        self._seed_company()
        self._seed_customer_type()
        self._seed_branch()
        self._seed_customer()
        self._seed_customer_address()
        self._seed_wrap_provider()
        self._seed_product_wrapper()
        self._seed_product_wrapper_type()
        self._seed_advisor()
        self._seed_account()
        self._seed_account_balance()
        self._seed_cash_movement()
        self._seed_transfer_request()
        self._seed_security_type()
        self._seed_security_subtype()
        self._seed_instrument()
        self._seed_order()
        self._seed_trade()
        self._seed_aggregate_order()
        self._seed_batch()
        self._seed_user_group()
        self._seed_application()
        self._seed_user()
        self._seed_user_group_membership()
        self._seed_session()
        counts = self._verify()
        print("[data_seeder] Complete.")
        return counts

    def _create_all_tables(self):
        stmts = [
            """CREATE TABLE IF NOT EXISTS Company (
                company_id INTEGER PRIMARY KEY, name VARCHAR, country VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS Branch (
                branch_id INTEGER PRIMARY KEY, company_id INTEGER,
                name VARCHAR, region VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS CustomerType (
                type_id INTEGER PRIMARY KEY, type_name VARCHAR, description VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS Customer (
                customer_id INTEGER PRIMARY KEY, name VARCHAR, type_id INTEGER,
                branch_id INTEGER, risk_rating VARCHAR, created_at VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS CustomerAddress (
                address_id INTEGER PRIMARY KEY, customer_id INTEGER,
                line1 VARCHAR, city VARCHAR, country VARCHAR, postcode VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS WrapProvider (
                provider_id INTEGER PRIMARY KEY, name VARCHAR, country VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS ProductWrapper (
                wrapper_id INTEGER PRIMARY KEY, provider_id INTEGER,
                wrapper_type VARCHAR, currency VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS ProductWrapperType (
                wrapper_type_id INTEGER PRIMARY KEY, name VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS Advisor (
                advisor_id INTEGER PRIMARY KEY, name VARCHAR,
                branch_id INTEGER, region VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS Account (
                account_id INTEGER PRIMARY KEY, customer_id INTEGER,
                advisor_id INTEGER, account_type VARCHAR, currency VARCHAR,
                opened_date DATE, wrapper_id INTEGER)""",
            """CREATE TABLE IF NOT EXISTS AccountBalance (
                balance_id INTEGER PRIMARY KEY, account_id INTEGER,
                balance DECIMAL, as_of_date DATE)""",
            """CREATE TABLE IF NOT EXISTS CashMovement (
                movement_id INTEGER PRIMARY KEY, account_id INTEGER,
                amount DECIMAL, direction VARCHAR, timestamp TIMESTAMP)""",
            """CREATE TABLE IF NOT EXISTS TransferRequest (
                transfer_id INTEGER PRIMARY KEY, source_account INTEGER,
                dest_account INTEGER, amount DECIMAL, status VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS SecurityType (
                security_type_id INTEGER PRIMARY KEY, type_name VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS SecuritySubType (
                subtype_id INTEGER PRIMARY KEY, security_type_id INTEGER, name VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS Instrument (
                instrument_id INTEGER PRIMARY KEY, name VARCHAR,
                security_type_id INTEGER, currency VARCHAR, subtype_id INTEGER)""",
            """CREATE TABLE IF NOT EXISTS "Order" (
                order_id INTEGER PRIMARY KEY, account_id INTEGER,
                instrument_id INTEGER, side VARCHAR, quantity DECIMAL)""",
            """CREATE TABLE IF NOT EXISTS Trade (
                trade_id INTEGER PRIMARY KEY, order_id INTEGER, account_id INTEGER,
                price DECIMAL, quantity DECIMAL, instrument_id INTEGER)""",
            """CREATE TABLE IF NOT EXISTS AggregateOrder (
                agg_order_id INTEGER PRIMARY KEY, created_at VARCHAR, order_id INTEGER)""",
            """CREATE TABLE IF NOT EXISTS Batch (
                batch_id INTEGER PRIMARY KEY, run_date DATE, agg_order_id INTEGER)""",
            """CREATE TABLE IF NOT EXISTS UserGroup (
                group_id INTEGER PRIMARY KEY, group_name VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS Application (
                app_id INTEGER PRIMARY KEY, app_name VARCHAR)""",
            """CREATE TABLE IF NOT EXISTS "User" (
                user_id INTEGER PRIMARY KEY, advisor_id INTEGER,
                name VARCHAR, group_id INTEGER)""",
            """CREATE TABLE IF NOT EXISTS UserGroupMembership (
                membership_id INTEGER PRIMARY KEY, user_id INTEGER, group_id INTEGER)""",
            """CREATE TABLE IF NOT EXISTS Session (
                session_id INTEGER PRIMARY KEY, user_id INTEGER,
                app_id INTEGER, login_time TIMESTAMP)""",
        ]
        for s in stmts:
            self.conn.execute(s)

    def _seed_company(self):
        rows = [
            (i + 1, COMPANY_NAMES[i], COUNTRIES[i]) for i in range(len(COMPANY_NAMES))
        ]
        for r in rows:
            self.company_ids.append(r[0])
        self.conn.executemany("INSERT INTO Company VALUES (?,?,?)", rows)

    def _seed_customer_type(self):
        rows = [(i + 1, CUST_TYPES[i], CUST_DESCS[i]) for i in range(len(CUST_TYPES))]
        for r in rows:
            self.customer_type_ids.append(r[0])
        self.conn.executemany("INSERT INTO CustomerType VALUES (?,?,?)", rows)

    def _seed_branch(self):
        rows = []
        for i in range(1, ROWS_MID + 1):
            city = BRANCH_CITIES[(i - 1) % len(BRANCH_CITIES)]
            rows.append(
                (
                    i,
                    random.choice(self.company_ids),
                    f"NatWest {city} Branch",
                    random.choice(REGIONS),
                )
            )
            self.branch_ids.append(i)
        self.conn.executemany("INSERT INTO Branch VALUES (?,?,?,?)", rows)

    def _seed_customer(self):
        rows = []
        for i in range(1, ROWS_LARGE + 1):
            rows.append(
                (
                    i,
                    CUSTOMER_NAMES[(i - 1) % len(CUSTOMER_NAMES)],
                    random.choice(self.customer_type_ids),
                    random.choice(self.branch_ids),
                    random.choice(RISK_RATINGS),
                    _ts(),
                )
            )
            self.customer_ids.append(i)
        self.conn.executemany("INSERT INTO Customer VALUES (?,?,?,?,?,?)", rows)

    def _seed_customer_address(self):
        rows = []
        for i, cid in enumerate(self.customer_ids, 1):
            city = BRANCH_CITIES[i % len(BRANCH_CITIES)]
            rows.append(
                (
                    i,
                    cid,
                    f"{random.randint(1, 200)} High Street",
                    city,
                    "United Kingdom",
                    f"{random.choice('ABCDEFGH')}{random.randint(1, 9)} "
                    f"{random.randint(1, 9)}{random.choice('ABCDEFGH')}{random.choice('ABCDEFGH')}",
                )
            )
        self.conn.executemany("INSERT INTO CustomerAddress VALUES (?,?,?,?,?,?)", rows)

    def _seed_wrap_provider(self):
        rows = [
            (i + 1, WRAP_PROVIDERS[i], "United Kingdom")
            for i in range(len(WRAP_PROVIDERS))
        ]
        for r in rows:
            self.wrap_provider_ids.append(r[0])
        self.conn.executemany("INSERT INTO WrapProvider VALUES (?,?,?)", rows)

    def _seed_product_wrapper(self):
        rows = []
        for i in range(1, ROWS_MID + 1):
            rows.append(
                (
                    i,
                    random.choice(self.wrap_provider_ids),
                    random.choice(WRAP_TYPES),
                    random.choice(CURRENCIES),
                )
            )
            self.wrapper_ids.append(i)
        self.conn.executemany("INSERT INTO ProductWrapper VALUES (?,?,?,?)", rows)

    def _seed_product_wrapper_type(self):
        rows = [(i + 1, WRAP_TYPES[i]) for i in range(len(WRAP_TYPES))]
        self.conn.executemany("INSERT INTO ProductWrapperType VALUES (?,?)", rows)

    def _seed_advisor(self):
        rows = []
        for i in range(1, ROWS_MID + 1):
            rows.append(
                (
                    i,
                    ADVISOR_NAMES[(i - 1) % len(ADVISOR_NAMES)],
                    random.choice(self.branch_ids),
                    random.choice(REGIONS),
                )
            )
            self.advisor_ids.append(i)
        self.conn.executemany("INSERT INTO Advisor VALUES (?,?,?,?)", rows)

    def _seed_account(self):
        rows = []
        for i in range(1, ROWS_CHILD + 1):
            rows.append(
                (
                    i,
                    random.choice(self.customer_ids),
                    random.choice(self.advisor_ids),
                    random.choice(ACCOUNT_TYPES),
                    random.choice(CURRENCIES),
                    _date_between(START, END),
                    random.choice(self.wrapper_ids),
                )
            )
            self.account_ids.append(i)
        self.conn.executemany("INSERT INTO Account VALUES (?,?,?,?,?,?,?)", rows)

    def _seed_account_balance(self):
        rows = [
            (i, acc, _amount(100, 500000), _date_between(START, END))
            for i, acc in enumerate(self.account_ids, 1)
        ]
        self.conn.executemany("INSERT INTO AccountBalance VALUES (?,?,?,?)", rows)

    def _seed_cash_movement(self):
        rows = []
        mid = 1
        for acc in self.account_ids:
            for _ in range(random.randint(1, 3)):
                rows.append(
                    (mid, acc, _amount(10, 50000), random.choice(DIRECTIONS), _ts())
                )
                mid += 1
        self.conn.executemany("INSERT INTO CashMovement VALUES (?,?,?,?,?)", rows)

    def _seed_transfer_request(self):
        rows = []
        for i in range(1, ROWS_CHILD + 1):
            src, dst = random.sample(self.account_ids, 2)
            rows.append(
                (i, src, dst, _amount(50, 100000), random.choice(TRANSFER_STATUS))
            )
        self.conn.executemany("INSERT INTO TransferRequest VALUES (?,?,?,?,?)", rows)

    def _seed_security_type(self):
        rows = [(i + 1, SECURITY_TYPES[i]) for i in range(len(SECURITY_TYPES))]
        for r in rows:
            self.security_type_ids.append(r[0])
        self.conn.executemany("INSERT INTO SecurityType VALUES (?,?)", rows)

    def _seed_security_subtype(self):
        rows = []
        sid = 1
        for type_id, tname in zip(self.security_type_ids, SECURITY_TYPES):
            for sub in SECURITY_SUBTYPES[tname]:
                rows.append((sid, type_id, sub))
                self.subtype_ids.append(sid)
                sid += 1
        self.conn.executemany("INSERT INTO SecuritySubType VALUES (?,?,?)", rows)

    def _seed_instrument(self):
        rows = []
        for i in range(1, ROWS_MID + 1):
            rows.append(
                (
                    i,
                    INSTRUMENT_NAMES[(i - 1) % len(INSTRUMENT_NAMES)],
                    random.choice(self.security_type_ids),
                    random.choice(CURRENCIES),
                    random.choice(self.subtype_ids),
                )
            )
            self.instrument_ids.append(i)
        self.conn.executemany("INSERT INTO Instrument VALUES (?,?,?,?,?)", rows)

    def _seed_order(self):
        rows = []
        for i in range(1, ROWS_LARGE + 1):
            rows.append(
                (
                    i,
                    random.choice(self.account_ids),
                    random.choice(self.instrument_ids),
                    random.choice(ORDER_SIDES),
                    round(random.uniform(1, 10000), 2),
                )
            )
            self.order_ids.append(i)
        self.conn.executemany('INSERT INTO "Order" VALUES (?,?,?,?,?)', rows)

    def _seed_trade(self):
        rows = []
        for i in range(1, ROWS_CHILD + 1):
            rows.append(
                (
                    i,
                    random.choice(self.order_ids),
                    random.choice(self.account_ids),
                    round(random.uniform(0.5, 500), 2),
                    round(random.uniform(1, 5000), 2),
                    random.choice(self.instrument_ids),
                )
            )
        self.conn.executemany("INSERT INTO Trade VALUES (?,?,?,?,?,?)", rows)

    def _seed_aggregate_order(self):
        rows = []
        for i in range(1, ROWS_MID + 1):
            rows.append((i, _ts(), self.order_ids[i - 1]))
            self.agg_order_ids.append(i)
        self.conn.executemany("INSERT INTO AggregateOrder VALUES (?,?,?)", rows)

    def _seed_batch(self):
        rows = [
            (i, _date_between(START, END), random.choice(self.agg_order_ids))
            for i in range(1, ROWS_SMALL * 2 + 1)
        ]
        self.conn.executemany("INSERT INTO Batch VALUES (?,?,?)", rows)

    def _seed_user_group(self):
        rows = [(i + 1, GROUP_NAMES[i]) for i in range(len(GROUP_NAMES))]
        for r in rows:
            self.group_ids.append(r[0])
        self.conn.executemany("INSERT INTO UserGroup VALUES (?,?)", rows)

    def _seed_application(self):
        rows = [(i + 1, APP_NAMES[i]) for i in range(len(APP_NAMES))]
        for r in rows:
            self.app_ids.append(r[0])
        self.conn.executemany("INSERT INTO Application VALUES (?,?)", rows)

    def _seed_user(self):
        rows = []
        for i in range(1, ROWS_MID + 1):
            rows.append(
                (
                    i,
                    random.choice(self.advisor_ids),
                    ADVISOR_NAMES[(i - 1) % len(ADVISOR_NAMES)] + " (User)",
                    random.choice(self.group_ids),
                )
            )
            self.user_ids.append(i)
        self.conn.executemany('INSERT INTO "User" VALUES (?,?,?,?)', rows)

    def _seed_user_group_membership(self):
        seen: Set[tuple] = set()
        rows = []
        mid = 1
        for uid in self.user_ids:
            gid = random.choice(self.group_ids)
            if (uid, gid) not in seen:
                seen.add((uid, gid))
                rows.append((mid, uid, gid))
                mid += 1
        self.conn.executemany("INSERT INTO UserGroupMembership VALUES (?,?,?)", rows)

    def _seed_session(self):
        rows = []
        sid = 1
        for uid in self.user_ids:
            for _ in range(random.randint(1, 4)):
                rows.append((sid, uid, random.choice(self.app_ids), _ts()))
                sid += 1
        self.conn.executemany("INSERT INTO Session VALUES (?,?,?,?)", rows)

    def _verify(self) -> Dict[str, int]:
        tables = [
            "Company",
            "Branch",
            "CustomerType",
            "Customer",
            "CustomerAddress",
            "WrapProvider",
            "ProductWrapper",
            "ProductWrapperType",
            "Advisor",
            "Account",
            "AccountBalance",
            "CashMovement",
            "TransferRequest",
            "SecurityType",
            "SecuritySubType",
            "Instrument",
            '"Order"',
            "Trade",
            "AggregateOrder",
            "Batch",
            "UserGroup",
            "Application",
            '"User"',
            "UserGroupMembership",
            "Session",
        ]
        counts = {}
        for t in tables:
            try:
                n = self.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                display = t.strip('"')
                counts[display] = n
                print(f"  {display:<25} {n:>5} rows")
            except Exception as e:
                print(f"  {t:<25} ERROR: {e}")
        return counts
