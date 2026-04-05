# Refactor TravelOpsEnv for Advanced RL Mechanics

This document outlines the proposed changes to `environment.py`, `models.py`, and `db_setup.py` to support tool resilience testing (chaos), RAG hybrid reasoning, and multi-turn Grader logic for a hackathon.

## User Review Required

> [!WARNING]
> Please review the changes proposed to the **Grader Logic**, specifically how we detect if an agent "gave up at 503" or "navigated pagination". I'll use the environment's action history to check if the agent retried `PROCESS_REFUND` and successfully passed cursors.

## Proposed Changes

---

### models.py

Update the `Action` model's action_type to match the new API structure.

#### [MODIFY] models.py
- **Remove:** `QUERY_DB`, `ISSUE_REFUND`
- **Add:** `SEARCH_BOOKINGS`, `FETCH_FLIGHT_STATUS`, `PROCESS_REFUND`, `SEARCH_POLICY_DOCS`, `END_EPISODE`
- Leave `MODIFY_BOOKING` and `SEND_REPLY`. Note that `SEND_REPLY` will no longer end the episode, which is why `END_EPISODE` is added.

---

### db_setup.py

Introduce properties required to test the Hard Task (fare classes and delay hours), and set up the scenario for the "RAG Policy Engine".

#### [MODIFY] db_setup.py
- Add `delay_hours INTEGER DEFAULT 0` to the `Flights` table schema.
- Add `fare_class TEXT DEFAULT 'STANDARD'` to the `Bookings` table schema.
- Create a new Hard Task user (`charlie`), who has:
  1. A Basic Economy (`BASIC`) booking.
  2. A flight that is delayed 4 hours (`delay_hours=4`).
  3. At least 3 bookings in total so that `SEARCH_BOOKINGS` (which has a limit of 2) requires the agent to use the pagination cursor to find the correct booking.

---

### environment.py

Modify the environment dynamics to match the new mock REST API APIs and robust grader.

#### [MODIFY] environment.py
- **`reset_hard()`:** Returns an email from `charlie` complaining about the delayed flight and asking for a refund for a specific booking.
- **`step()` adjustments:**
  - `SEARCH_BOOKINGS`: Queries the DB for a given `user_id` with `LIMIT 2 OFFSET {cursor}`. Returns `next_cursor` if there are more records.
  - `FETCH_FLIGHT_STATUS`: Looks up flight details in the database by `flight_id` (returns `delay_hours`).
  - `PROCESS_REFUND`: Has a 20% chance of returning a 503 error `{"status": 503, "message": "Payment Gateway Timeout"}`. If successful, creates the refund effect. Emits the outcome in observation so the agent handles it.
  - `SEARCH_POLICY_DOCS`: Contains a hardcoded dictionary mapping keywords to policy text. Includes the hidden rule: `"Flights delayed > 3 hours are eligible for 100% refund, superseding the 'Basic Economy' non-refundable rule."`
  - `SEND_REPLY`: Appends a reply to the customer but does *not* set `done = True`.
  - `END_EPISODE`: Ends the episode.
- **`grade_hard_task()`:** Sweeps the `action_history` to compute a grade based on the multi-objective prompt:
  - Did the agent retry after a 503?
  - Did the agent use pagination (`cursor > 0`)?
  - Did the agent check policy docs?
  - Score mapping:
    - `1.0`: Successfully bypassed 503, used pagination, checked policy, and issued refund.
    - `0.0`: Gave up at 503 (meaning 503 was logged but the booking stat remained un-refunded).
    - `-0.5`: Ignored the policy and denied the refund (completed the episode without issuing refund).

## Open Questions

> [!IMPORTANT]
> Is the addition of `END_EPISODE` to the action types acceptable, or would you prefer the agent to pass a flag like `is_final: True` inside the payload of `SEND_REPLY` to end the conversation? Both work, but `END_EPISODE` avoids overloading `SEND_REPLY`.

## Verification Plan

### Automated Tests
1. Script an agent trajectory simulating correct behavior:
   - Search bookings (page 1)
   - Search bookings (page 2 with cursor)
   - Fetch flight status (see 4 hr delay)
   - Search policy doc (find 3 hr override)
   - Process refund (hit 503)
   - Process refund again (success)
   - Send reply (apology and refund confirmation)
   - End episode.
2. Script alternative trajectories (e.g., getting the 503 and ending the episode, or denying the refund) to ensure the Grader outputs `0.0` or `-0.5` respectively.
