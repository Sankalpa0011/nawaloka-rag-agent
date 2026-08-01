"""Post-import integrity check for the ported database.

Run this immediately after `docker compose up`. It fails loudly rather than
letting a partially-imported database sit there looking fine - a silently
truncated import is the kind of thing you only notice when the agent starts
answering questions wrongly three days later.

Beyond row counts it asserts that each known data hazard is actually defused by
the views in db/init/003_views_indexes.sql, so a regression in those views is
caught here rather than in an eval run.

Usage:
    python scripts/verify_db.py            # uses DATABASE_URL from .env
    python scripts/verify_db.py --ro       # also checks the read-only role
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running as a bare script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

try:
    import psycopg
except ImportError:
    sys.exit("psycopg is not installed. Run: pip install -r backend/requirements.txt")

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_URL = "postgresql://nawaloka:nawaloka_dev_pw@localhost:5433/nawaloka"
DEFAULT_RO_URL = "postgresql://chatbot_ro:chatbot_ro_dev_pw@localhost:5433/nawaloka"

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"

EXPECTED_COUNTS = {
    "specialties": 10,
    "doctors": 11,
    "channeling_sessions": 15,
    "lab_tests": 17,
    "health_packages": 11,
}

failures: list[str] = []
passes = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global passes
    if ok:
        passes += 1
        print(f"  {GREEN}PASS{RESET}  {label}" + (f"  {DIM}{detail}{RESET}" if detail else ""))
    else:
        failures.append(label)
        print(f"  {RED}FAIL{RESET}  {label}" + (f"  {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{YELLOW}{title}{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("DATABASE_URL", DEFAULT_URL))
    parser.add_argument("--ro-url", default=os.getenv("DATABASE_RO_URL", DEFAULT_RO_URL))
    parser.add_argument("--ro", action="store_true", help="also verify the read-only role")
    args = parser.parse_args()

    print(f"{DIM}Connecting to {args.url.rsplit('@', 1)[-1]}{RESET}")

    with psycopg.connect(args.url, connect_timeout=10) as conn, conn.cursor() as cur:

        # ------------------------------------------------------------------
        section("Row counts")
        for table, expected in EXPECTED_COUNTS.items():
            cur.execute(f"SELECT count(*) FROM {table}")
            actual = cur.fetchone()[0]
            check(f"{table} has {expected} rows", actual == expected, f"got {actual}")

        # ------------------------------------------------------------------
        section("Schema objects")
        for view in ("v_sessions_expanded", "v_doctor_directory", "v_package_services"):
            cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{view}",))
            check(f"view {view} exists", cur.fetchone()[0])

        cur.execute("SELECT count(*) FROM pg_extension WHERE extname = 'pg_trgm'")
        check("pg_trgm extension installed", cur.fetchone()[0] == 1)

        cur.execute("SELECT count(*) FROM pg_indexes WHERE schemaname='public' AND indexname LIKE '%_trgm'")
        n_trgm = cur.fetchone()[0]
        check("trigram indexes created", n_trgm >= 8, f"found {n_trgm}")

        # ------------------------------------------------------------------
        section("Identity sequences re-synced (TRAP 5: ids start at 9 and 5)")
        for table in EXPECTED_COUNTS:
            cur.execute(
                "SELECT last_value, (SELECT max(id) FROM %s) FROM pg_sequences "
                "WHERE schemaname='public' AND sequencename = %%s" % table,
                (f"{table}_id_seq",),
            )
            row = cur.fetchone()
            if row is None:
                check(f"{table} sequence exists", False, "sequence not found")
                continue
            last_value, max_id = row
            check(f"{table} sequence at max(id)={max_id}", last_value == max_id,
                  f"sequence is at {last_value}")

        # ------------------------------------------------------------------
        section("TRAP 1: day_of_week = 'Daily' must match every weekday")
        cur.execute("SELECT count(*) FROM channeling_sessions WHERE lower(btrim(day_of_week)) = 'daily'")
        check("source data still contains a literal 'Daily' row", cur.fetchone()[0] == 1)

        cur.execute("""
            SELECT doctor_name FROM v_sessions_expanded
            WHERE weekday = 'Tuesday' AND is_recurring_daily
        """)
        daily_docs = [r[0] for r in cur.fetchall()]
        check("'Daily' doctor appears in a Tuesday query", "Dr. Duminda Pathirana" in daily_docs,
              f"got {daily_docs}")

        cur.execute("""
            SELECT count(DISTINCT weekday) FROM v_sessions_expanded
            WHERE doctor_name = 'Dr. Duminda Pathirana'
        """)
        check("'Daily' expands to all 7 weekdays", cur.fetchone()[0] == 7)

        cur.execute("SELECT count(*) FROM v_sessions_expanded")
        # 14 single-day sessions + 1 'Daily' session expanded to 7 = 21
        check("v_sessions_expanded row count", cur.fetchone()[0] == 21)

        # ------------------------------------------------------------------
        section("TRAP 2: doctors with no channeling sessions must survive")
        cur.execute("SELECT count(*) FROM v_doctor_directory")
        check("directory lists all 11 doctors", cur.fetchone()[0] == 11)

        cur.execute("""
            SELECT doctor_id, doctor_name FROM v_doctor_directory
            WHERE NOT has_published_sessions ORDER BY doctor_id
        """)
        no_sessions = cur.fetchall()
        check("exactly 2 doctors flagged as having no sessions", len(no_sessions) == 2,
              f"got {no_sessions}")
        check("they are doctors 8 and 9", [d[0] for d in no_sessions] == [8, 9],
              f"got {[d[0] for d in no_sessions]}")

        # ------------------------------------------------------------------
        section("TRAP 3: package contents are free text, not joinable to lab_tests")
        cur.execute("""
            SELECT DISTINCT package_name FROM v_package_services
            WHERE service_item ILIKE '%pap smear%' ORDER BY package_name
        """)
        pap = [r[0] for r in cur.fetchall()]
        check("'Pap Smear' resolves to 3 packages", len(pap) == 3, f"got {pap}")

        cur.execute("""
            SELECT count(*) FROM v_package_services vps
            JOIN lab_tests lt ON lower(btrim(lt.test_name)) = lower(btrim(vps.service_item))
        """)
        joinable = cur.fetchone()[0]
        check("package text does NOT join cleanly to lab_tests (join is a trap)",
              joinable == 0, f"{joinable} rows joined - the trap assumption changed")

        cur.execute("SELECT count(*) FROM v_package_services")
        n_services = cur.fetchone()[0]
        check("package services exploded into rows", n_services >= 70, f"got {n_services}")

        # ------------------------------------------------------------------
        section("TRAP 4: honorifics break exact name matching")
        cur.execute("SELECT name_plain FROM v_doctor_directory WHERE doctor_id = 10")
        check("'Prof. Arjuna De Silva' -> 'Arjuna De Silva'",
              cur.fetchone()[0] == "Arjuna De Silva")

        cur.execute("""
            SELECT doctor_name, consultation_fee FROM v_doctor_directory
            WHERE similarity(name_plain, 'De Silva') > 0.3
            ORDER BY similarity(name_plain, 'De Silva') DESC LIMIT 1
        """)
        row = cur.fetchone()
        check("fuzzy surname lookup finds Prof. De Silva's fee",
              row is not None and float(row[1]) == 4200.00, f"got {row}")

        # ------------------------------------------------------------------
        section("TRAP 6: room_number overflows the declared VARCHAR(20)")
        cur.execute("SELECT max(length(room_number)) FROM channeling_sessions")
        maxlen = cur.fetchone()[0]
        check("longest room_number stored without truncation", maxlen == 26,
              f"got {maxlen} (MySQL would silently truncate this to 20)")

        cur.execute("""
            SELECT count(*) FROM channeling_sessions
            WHERE room_number = 'Room 004 (Paediatric Wing)'
        """)
        check("'Room 004 (Paediatric Wing)' intact", cur.fetchone()[0] == 1)

        # ------------------------------------------------------------------
        section("Referential integrity")
        cur.execute("SELECT count(*) FROM doctors d LEFT JOIN specialties s ON s.id=d.specialty_id WHERE s.id IS NULL")
        check("every doctor has a valid specialty", cur.fetchone()[0] == 0)
        cur.execute("SELECT count(*) FROM channeling_sessions c LEFT JOIN doctors d ON d.id=c.doctor_id WHERE d.id IS NULL")
        check("every session has a valid doctor", cur.fetchone()[0] == 0)

    # ----------------------------------------------------------------------
    if args.ro:
        section("Read-only role (defence in depth for the SQL agent)")
        try:
            with psycopg.connect(args.ro_url, connect_timeout=10, autocommit=True) as ro:
                with ro.cursor() as cur:
                    cur.execute("SELECT count(*) FROM doctors")
                    check("chatbot_ro can read doctors", cur.fetchone()[0] == 11)

                for stmt, label in [
                    ("DELETE FROM doctors WHERE id = 1", "DELETE is rejected"),
                    ("UPDATE doctors SET consultation_fee = 0", "UPDATE is rejected"),
                    ("CREATE TABLE evil (x int)", "CREATE TABLE is rejected"),
                ]:
                    try:
                        with ro.cursor() as cur:
                            cur.execute(stmt)
                        check(label, False, "statement SUCCEEDED - role is not read-only")
                    except psycopg.Error as exc:
                        check(label, True, type(exc).__name__)

                for tbl in ("routing_decisions", "user_memory"):
                    try:
                        with ro.cursor() as cur:
                            cur.execute(f"SELECT * FROM {tbl} LIMIT 1")
                        check(f"chatbot_ro cannot read {tbl}", False, "SELECT succeeded")
                    except psycopg.Error:
                        check(f"chatbot_ro cannot read {tbl}", True, "permission denied")
        except psycopg.OperationalError as exc:
            check("connect as chatbot_ro", False, str(exc).splitlines()[0])

    # ----------------------------------------------------------------------
    print()
    if failures:
        print(f"{RED}{len(failures)} check(s) FAILED{RESET} ({passes} passed):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"{GREEN}All {passes} checks passed.{RESET} Database import verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
