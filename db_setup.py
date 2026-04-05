import sqlite3
import random
import string


def _rand_booking_id():
    return f"BKG-{random.randint(100000, 999999)}"


def _rand_flight_id():
    return f"FL{random.randint(100000, 999999)}"


def _rand_amount():
    return random.choice([1200, 1800, 2500, 3200, 3800, 4500, 5200, 6000, 7500, 8900])


def _rand_name():
    first = random.choice([
        "Liam", "Emma", "Noah", "Olivia", "James", "Sophia", "Mason",
        "Ava", "Ethan", "Mia", "Lucas", "Harper", "Logan", "Ella",
        "Aiden", "Isla", "Jack", "Aria", "Owen", "Chloe",
    ])
    last = random.choice([
        "Smith", "Johnson", "Brown", "Davis", "Wilson", "Moore", "Taylor",
        "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Lee",
    ])
    return f"{first} {last}"


def _rand_city():
    return random.choice([
        "Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata",
        "Hyderabad", "Pune", "Jaipur", "Lucknow", "Goa",
        "Ahmedabad", "Kochi", "Varanasi", "Udaipur", "Srinagar",
    ])


def _rand_time(hour_low=6, hour_high=23):
    h = random.randint(hour_low, hour_high)
    m = random.choice([0, 15, 30, 45])
    return f"{h:02d}:{m:02d}"


def setup_database():
    """
    Creates an in-memory SQLite database with randomised core IDs and
    20 distractor rows per table.  Returns (connection, test_data_dict).
    """
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = conn.cursor()

    # ── Schema ────────────────────────────────────────────────────────
    cursor.execute("""
        CREATE TABLE Users (
            user_id   INTEGER PRIMARY KEY,
            name      TEXT NOT NULL,
            email     TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE Flights (
            flight_id      TEXT PRIMARY KEY,
            origin         TEXT NOT NULL,
            destination    TEXT NOT NULL,
            departure_time TEXT NOT NULL,
            arrival_time   TEXT NOT NULL,
            status         TEXT NOT NULL,
            delay_hours    INTEGER DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE Bookings (
            booking_id TEXT PRIMARY KEY,
            user_id    INTEGER,
            flight_id  TEXT,
            amount     INTEGER,
            status     TEXT,
            fare_class TEXT DEFAULT 'STANDARD',
            FOREIGN KEY(user_id)   REFERENCES Users(user_id),
            FOREIGN KEY(flight_id) REFERENCES Flights(flight_id)
        )
    """)

    # ── Randomised core test data ─────────────────────────────────────
    bob_booking_id = _rand_booking_id()
    bob_flight_id  = _rand_flight_id()
    bob_amount     = _rand_amount()

    alice_booking_id = _rand_booking_id()
    while alice_booking_id == bob_booking_id:
        alice_booking_id = _rand_booking_id()
    alice_flight_id = _rand_flight_id()
    while alice_flight_id == bob_flight_id:
        alice_flight_id = _rand_flight_id()

    charlie_booking_id = _rand_booking_id()
    while charlie_booking_id in [bob_booking_id, alice_booking_id]:
        charlie_booking_id = _rand_booking_id()
    charlie_flight_id = _rand_flight_id()
    while charlie_flight_id in [bob_flight_id, alice_flight_id]:
        charlie_flight_id = _rand_flight_id()

    # Bob – User 102, confirmed booking on a confirmed flight
    cursor.execute(
        "INSERT INTO Users VALUES (102, 'Bob', 'bob@example.com')")
    cursor.execute(
        "INSERT INTO Flights (flight_id, origin, destination, departure_time, arrival_time, status) VALUES (?, 'Delhi', 'Goa', '09:00', '12:00', 'CONFIRMED')",
        (bob_flight_id,))
    cursor.execute(
        "INSERT INTO Bookings (booking_id, user_id, flight_id, amount, status) VALUES (?, 102, ?, ?, 'CONFIRMED')",
        (bob_booking_id, bob_flight_id, bob_amount))

    # Alice – User 101, confirmed booking on a CANCELLED flight
    cursor.execute(
        "INSERT INTO Users VALUES (101, 'Alice', 'alice@example.com')")
    cursor.execute(
        "INSERT INTO Flights (flight_id, origin, destination, departure_time, arrival_time, status) VALUES (?, 'Mumbai', 'Bangalore', '10:30', '13:00', 'CANCELLED')",
        (alice_flight_id,))
    cursor.execute(
        "INSERT INTO Bookings (booking_id, user_id, flight_id, amount, status) VALUES (?, 101, ?, 3500, 'CONFIRMED')",
        (alice_booking_id, alice_flight_id))

    # Charlie - User 103, Hard Task
    # Basic Economy booking, flight delayed 4 hrs, also user 103 needs 3 total bookings to require pagination
    cursor.execute(
        "INSERT INTO Users VALUES (103, 'Charlie', 'charlie@example.com')")
    cursor.execute(
        "INSERT INTO Flights (flight_id, origin, destination, departure_time, arrival_time, status, delay_hours) VALUES (?, 'Kolkata', 'Delhi', '12:00', '14:30', 'DELAYED', 4)",
        (charlie_flight_id,))
    cursor.execute(
        "INSERT INTO Bookings (booking_id, user_id, flight_id, amount, status, fare_class) VALUES (?, 103, ?, 2200, 'CONFIRMED', 'BASIC')",
        (charlie_booking_id, charlie_flight_id))

    # Charlie's distractors so pagination is tested
    for _ in range(2):
        dist_bkg = _rand_booking_id()
        while dist_bkg in [bob_booking_id, alice_booking_id, charlie_booking_id]:
           dist_bkg = _rand_booking_id()
        dist_flt = _rand_flight_id()
        cursor.execute(
            "INSERT INTO Flights (flight_id, origin, destination, departure_time, arrival_time, status) VALUES (?, 'CityA', 'CityB', '00:00', '02:00', 'CONFIRMED')",
            (dist_flt,))
        cursor.execute(
            "INSERT INTO Bookings (booking_id, user_id, flight_id, amount, status, fare_class) VALUES (?, 103, ?, 1800, 'CONFIRMED', 'STANDARD')",
            (dist_bkg, dist_flt))


    # ── Alternative flights for the hard task (Alice rescheduling) ────
    alt_flights = []
    
    # 3 valid alternatives (arrive before 18:00, same route)
    for i in range(3):
        fid = _rand_flight_id()
        while fid in [bob_flight_id, alice_flight_id, charlie_flight_id] + [f[0] for f in alt_flights]:
            fid = _rand_flight_id()
        arr = _rand_time(hour_low=13, hour_high=17)
        dep = _rand_time(hour_low=9, hour_high=12)
        alt_flights.append((fid, "Mumbai", "Bangalore", dep, arr, "CONFIRMED"))

    # 2 invalid alternatives (arrive after 18:00, same route)
    for i in range(2):
        fid = _rand_flight_id()
        while fid in [bob_flight_id, alice_flight_id, charlie_flight_id] + [f[0] for f in alt_flights]:
            fid = _rand_flight_id()
        arr = _rand_time(hour_low=18, hour_high=23)
        dep = _rand_time(hour_low=14, hour_high=17)
        alt_flights.append((fid, "Mumbai", "Bangalore", dep, arr, "CONFIRMED"))

    for af in alt_flights:
        cursor.execute("INSERT INTO Flights (flight_id, origin, destination, departure_time, arrival_time, status) VALUES (?, ?, ?, ?, ?, ?)", af)

    valid_alt_flight_ids = [f[0] for f in alt_flights[:3]]

    # ── 20 distractor users ───────────────────────────────────────────
    used_user_ids = {101, 102, 103}
    distractor_user_ids = []
    while len(distractor_user_ids) < 20:
        uid = random.randint(200, 999)
        if uid not in used_user_ids:
            used_user_ids.add(uid)
            distractor_user_ids.append(uid)

    for uid in distractor_user_ids:
        name = _rand_name()
        email = f"{name.split()[0].lower()}{uid}@example.com"
        cursor.execute("INSERT INTO Users VALUES (?, ?, ?)", (uid, name, email))

    # ── 20 distractor flights ─────────────────────────────────────────
    used_flight_ids = {bob_flight_id, alice_flight_id, charlie_flight_id} | {f[0] for f in alt_flights}
    distractor_flight_ids = []
    while len(distractor_flight_ids) < 20:
        fid = _rand_flight_id()
        if fid not in used_flight_ids:
            used_flight_ids.add(fid)
            distractor_flight_ids.append(fid)

    for fid in distractor_flight_ids:
        origin = _rand_city()
        dest = _rand_city()
        while dest == origin:
            dest = _rand_city()
        dep = _rand_time()
        arr = _rand_time()
        status = random.choice(["CONFIRMED", "CONFIRMED", "CONFIRMED", "DELAYED", "CANCELLED"])
        cursor.execute("INSERT INTO Flights (flight_id, origin, destination, departure_time, arrival_time, status) VALUES (?, ?, ?, ?, ?, ?)",
                       (fid, origin, dest, dep, arr, status))

    # ── 20 distractor bookings ────────────────────────────────────────
    used_booking_ids = {bob_booking_id, alice_booking_id, charlie_booking_id}
    for _ in range(20):
        bid = _rand_booking_id()
        while bid in used_booking_ids:
            bid = _rand_booking_id()
        used_booking_ids.add(bid)

        uid = random.choice(distractor_user_ids)
        fid = random.choice(distractor_flight_ids)
        amt = _rand_amount()
        status = random.choice(["CONFIRMED", "CONFIRMED", "CANCELLED", "PENDING"])
        fc = random.choice(["STANDARD", "BASIC", "PREMIUM"])
        cursor.execute("INSERT INTO Bookings (booking_id, user_id, flight_id, amount, status, fare_class) VALUES (?, ?, ?, ?, ?, ?)",
                       (bid, uid, fid, amt, status, fc))

    conn.commit()

    # ── Return connection + lookup dict for environment & grader ──────
    test_data = {
        "bob": {
            "user_id": 102,
            "booking_id": bob_booking_id,
            "flight_id": bob_flight_id,
            "amount": bob_amount,
        },
        "alice": {
            "user_id": 101,
            "booking_id": alice_booking_id,
            "flight_id": alice_flight_id,
        },
        "charlie": {
            "user_id": 103,
            "booking_id": charlie_booking_id,
            "flight_id": charlie_flight_id,
        },
        "valid_alt_flight_ids": valid_alt_flight_ids,
        "time_constraint": "18:00",
    }
    return conn, test_data


if __name__ == "__main__":
    conn, td = setup_database()
    c = conn.cursor()

    print("=== TEST DATA ===")
    for k, v in td.items():
        print(f"  {k}: {v}")

    print("\n=== Users (sample) ===")
    for row in c.execute("SELECT * FROM Users LIMIT 5"):
        print(" ", row)

    print(f"\n=== Total Users:    {c.execute('SELECT count(*) FROM Users').fetchone()[0]}")
    print(f"=== Total Flights:  {c.execute('SELECT count(*) FROM Flights').fetchone()[0]}")
    print(f"=== Total Bookings: {c.execute('SELECT count(*) FROM Bookings').fetchone()[0]}")
