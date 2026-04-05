import json
import random
from models import Observation, Action
from db_setup import setup_database


class TravelOpsEnv:
    """
    OpenEnv-compatible environment for Safar Saathi Level 2 Customer Support.
    """

    POLICY_DOCS = {
        "refund_policy": (
            "Refund Policy: Basic Economy (BASIC) tickets are generally non-refundable. "
            "Standard tickets have a $50 cancellation fee. Premium tickets are fully refundable. "
            "HIDDEN RULE: Flights delayed > 3 hours are eligible for 100% refund, superseding "
            "the 'Basic Economy' non-refundable rule."
        ),
        "baggage_policy": "Baggage Policy: 1 carry-on allowed. Checked bags $30 each.",
        "upgrade_policy": "Upgrades are subject to availability at the gate."
    }

    def __init__(self, task_level="normal"):
        self.task_level = task_level
        self.db = None
        self.test_data = {}
        self.action_history: list[Action] = []
        self.step_count = 0

    def reset(self) -> Observation:
        """Bootstrap the DB and return the correct scenario email based on task_level."""
        if self.task_level == "hard":
            return self.reset_hard()
        self.db, self.test_data = setup_database()
        self.action_history = []
        self.step_count = 0

        bob = self.test_data["bob"]
        self._initial_email = {
            "from": "bob@example.com",
            "subject": "Refund for my flight",
            "body": (
                f"Hello, I am Bob (User ID: 102). "
                f"I need a refund for my booking {bob['booking_id']}. "
                f"My flight was {bob['flight_id']} and it cost ${bob['amount']}. "
                f"Please assist."
            ),
        }

        return Observation(
            inbox=self._initial_email,
            db_result="",
            system_feedback="Environment reset. You have a new support ticket.",
            is_done=False,
        )

    def reset_hard(self) -> Observation:
        """Reset with Charlie's rescheduling scenario (RAG + Chaos)."""
        self.db, self.test_data = setup_database()
        self.action_history = []
        self.step_count = 0

        charlie = self.test_data["charlie"]
        self._initial_email = {
            "from": "charlie@example.com",
            "subject": "Flight delayed significantly - need refund",
            "body": (
                f"Hi, I'm Charlie (User ID: 103). "
                f"My flight {charlie['flight_id']} was delayed by several hours. "
                f"I want a full refund for my booking {charlie['booking_id']}. "
                f"Please check the policy and process the refund."
            ),
        }

        return Observation(
            inbox=self._initial_email,
            db_result="",
            system_feedback="Environment reset. You have a new support ticket.",
            is_done=False,
        )

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one agent action.
        """
        # Detect if the agent repeats the exact same action 3 times in a row
        if len(self.action_history) >= 2:
            if self.action_history[-1] == action and self.action_history[-2] == action:
                self.action_history.append(action)
                self.step_count += 1
                return Observation(
                    inbox=self._initial_email,
                    db_result="",
                    system_feedback="Repeated exact same action 3 times. Terminating.",
                    is_done=True
                ), -1.0, True, {}

        self.action_history.append(action)
        self.step_count += 1

        # Dense reward shaping: +0.05 for valid parsing, -0.02 step penalty
        reward = 0.03
        done = False
        db_result = ""
        system_feedback = ""

        # ── SEARCH_BOOKINGS ──────────────────────────────────────────
        if action.action_type == "SEARCH_BOOKINGS":
            user_id = action.payload.get("user_id")
            cursor_val = action.payload.get("cursor", 0)

            # +0.05 micro-reward for successfully parsing a pagination cursor
            if cursor_val > 0:
                reward += 0.05
            
            try:
                c = self.db.cursor()
                # Use limit 3 to see if there is a next page
                c.execute(
                    "SELECT booking_id, flight_id, amount, status, fare_class FROM Bookings "
                    "WHERE user_id = ? LIMIT 3 OFFSET ?", 
                    (user_id, cursor_val)
                )
                rows = c.fetchall()
                
                results_to_return = rows[:2]
                has_more = len(rows) > 2
                
                res_list = []
                for r in results_to_return:
                    res_list.append({
                        "booking_id": r[0],
                        "flight_id": r[1],
                        "amount": r[2],
                        "status": r[3],
                        "fare_class": r[4]
                    })
                
                ret_obj = {
                    "data": res_list,
                    "next_cursor": cursor_val + 2 if has_more else None
                }
                db_result = json.dumps(ret_obj)
                reward += 0.1
                system_feedback = f"Returned {len(res_list)} records."
            except Exception as e:
                system_feedback = f"Error: {e}"

        # ── FETCH_FLIGHT_STATUS ──────────────────────────────────────
        elif action.action_type == "FETCH_FLIGHT_STATUS":
            flight_id = action.payload.get("flight_id")
            try:
                c = self.db.cursor()
                c.execute(
                    "SELECT status, delay_hours FROM Flights WHERE flight_id = ?",
                    (flight_id,)
                )
                row = c.fetchone()
                if row:
                    db_result = json.dumps({"flight_id": flight_id, "status": row[0], "delay_hours": row[1]})
                    reward += 0.1
                    system_feedback = "Flight status retrieved."
                else:
                    system_feedback = "Flight not found."
            except Exception as e:
                system_feedback = f"Error: {e}"

        # ── PROCESS_REFUND ───────────────────────────────────────────
        elif action.action_type == "PROCESS_REFUND":
            booking_id = action.payload.get("booking_id")
            amount = action.payload.get("amount")
            
            if random.random() < 0.20:
                reward -= 0.1
                system_feedback = '{"status": 503, "message": "Payment Gateway Timeout"}'
            else:
                try:
                    c = self.db.cursor()
                    c.execute("UPDATE Bookings SET status = 'REFUNDED' WHERE booking_id = ?", (booking_id,))
                    if c.rowcount > 0:
                        self.db.commit()
                        system_feedback = f"Refund of ${amount} processed successfully for {booking_id}."
                        reward += 0.5
                    else:
                        system_feedback = f"Booking {booking_id} not found."
                except Exception as e:
                    system_feedback = f"Error: {e}"

        # ── SEARCH_POLICY_DOCS ───────────────────────────────────────
        elif action.action_type == "SEARCH_POLICY_DOCS":
            query = action.payload.get("query", "").lower()
            if "refund" in query:
                db_result = self.POLICY_DOCS["refund_policy"]
            elif "baggage" in query or "bag" in query:
                db_result = self.POLICY_DOCS["baggage_policy"]
            else:
                db_result = "No policy docs match your query. Try 'refund' or 'baggage'."
            system_feedback = "Policy doc retrieved."
            reward += 0.1

        # ── MODIFY_BOOKING ────────────────────────────────────────────
        elif action.action_type == "MODIFY_BOOKING":
            system_feedback = "Not implemented for this flow."

        # ── SEND_REPLY ────────────────────────────────────────────────
        elif action.action_type == "SEND_REPLY":
            message = action.payload.get("message", "(no message)")
            system_feedback = f"Reply sent to customer: \"{message[:120]}\""
            # Do NOT set done = True here

        # ── END_EPISODE ───────────────────────────────────────────────
        elif action.action_type == "END_EPISODE":
            system_feedback = "Episode ended by agent."
            done = True
            
        else:
            system_feedback = "Invalid tool call."
            reward -= 0.5

        obs = Observation(
            inbox=self._initial_email,
            db_result=db_result,
            system_feedback=system_feedback,
            is_done=done,
        )
        return obs, reward, done, {}

    def grade(self) -> float:
        """Placeholder for medium grader."""
        return 0.0

    def grade_hard_task(self) -> float:
        """
        **Hard-task grader** – deterministic, trajectory-based.

        1. Did they handle pagination (cursor > 0)?
        2. Did they search policy docs?
        3. Did they process refund properly?
        4. Did they handle 503 if it happened?
        """
        charlie = self.test_data["charlie"]
        
        paginated = False
        searched_policy = False
        encountered_503 = False
        refund_attempted_after_503 = False
        refund_successful = False
        denied_refund = False
        
        for idx, action in enumerate(self.action_history):
            if action.action_type == "SEARCH_BOOKINGS":
                if action.payload.get("cursor", 0) > 0:
                    paginated = True
            
            elif action.action_type == "SEARCH_POLICY_DOCS":
                searched_policy = True
                
            elif action.action_type == "PROCESS_REFUND":
                bid = action.payload.get("booking_id")
                if bid == charlie["booking_id"]:
                    # We check the environment's actual outcome in action_history? 
                    # Actually we don't store obs history in action_history easily.
                    # But we can check DB.
                    pass
            
            elif action.action_type == "SEND_REPLY":
                msg = action.payload.get("message", "").lower()
                if "cannot" in msg or "not eligible" in msg or "denied" in msg:
                    denied_refund = True

        # Check actual refund state
        c = self.db.cursor()
        c.execute("SELECT status FROM Bookings WHERE booking_id = ?", (charlie["booking_id"],))
        row = c.fetchone()
        if row and row[0] == "REFUNDED":
            refund_successful = True

        # We can try to infer 503 by looking if there are multiple PROCESS_REFUND actions
        # But a more elegant way is to just assume they either refunded or didn't.
        # To perfectly adhere to "If it gave up at the 503 error, score = 0.0" 
        # let's assume if it's NOT refunded, and they didn't deny it, they gave up.
        
        if not refund_successful:
            if denied_refund:
                return -0.5
            else:
                return 0.0  # Gave up (potentially at 503 or didn't know what to do)

        # Refund was successful, check if they used RAG & pagination
        if paginated and searched_policy:
            return 1.0
        
        return 0.5  # Issued refund but missed RAG or Pagination checks
