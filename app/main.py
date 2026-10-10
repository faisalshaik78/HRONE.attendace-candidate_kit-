"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience)
"""

import math
import os
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo import MongoClient

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
PRESENT_STATUSES = {"PRESENT", "WFH", "ON_DUTY"}
ALL_STATUSES = {"PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"}


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0")


def round_half_up(value: float, digits: int) -> float:
    factor = 10 ** digits
    return float(math.floor(value * factor + 0.5) / factor)


def parse_hhmm(value: str) -> time:
    hours, minutes = map(int, value.split(":"))
    return time(hour=hours, minute=minutes)


def _as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _to_epoch_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    return int(_as_utc(dt).timestamp() * 1000)


def _to_epoch_ms_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return _to_epoch_ms(value)
    if isinstance(value, list):
        return [_to_epoch_ms(v) if isinstance(v, datetime) else v for v in value]
    if isinstance(value, dict):
        return {k: _to_epoch_ms_value(v) for k, v in value.items()}
    return value


def _convert_history_to_api(entry: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "at": _to_epoch_ms(entry.get("at")),
        "by": entry.get("by"),
        "reason": entry.get("reason"),
        "changes": {},
    }
    for key, value in (entry.get("changes") or {}).items():
        out["changes"][key] = {
            "from": _to_epoch_ms_value(value.get("from")) if "from" in value else None,
            "to": _to_epoch_ms_value(value.get("to")) if "to" in value else None,
        }
    return out


def _serialize_employee(doc: Dict[str, Any]) -> Dict[str, Any]:
    api_doc = {k: v for k, v in doc.items() if k != "_id"}
    if "created_at" in api_doc and api_doc["created_at"] is not None:
        api_doc["created_at"] = _to_epoch_ms(api_doc["created_at"])
    return api_doc


def _serialize_record(doc: Dict[str, Any]) -> Dict[str, Any]:
    api_doc = {k: v for k, v in doc.items() if k != "_id"}
    for field in ("punch_in", "punch_out"):
        if field in api_doc:
            api_doc[field] = _to_epoch_ms(api_doc[field])
    if "history" in api_doc:
        api_doc["history"] = [_convert_history_to_api(item) for item in api_doc["history"]]
    return api_doc


def ensure_indexes() -> None:
    try:
        db.employees.create_index("emp_code", unique=True)
        db.employees.create_index("department")
        db.employees.create_index("joined_on")
        db.attendance_logs.create_index([("emp_code", 1), ("date", 1)], unique=True)
        db.attendance_logs.create_index([("date", -1), ("emp_code", 1)])
        db.attendance_logs.create_index([("emp_code", 1), ("date", -1)])
        db.attendance_logs.create_index([("status", 1), ("date", 1)])
        db.attendance_logs.create_index([("date", 1), ("late_minutes", 1)])
    except Exception:
        pass


@app.on_event("startup")
def startup_event() -> None:
    try:
        ensure_indexes()
    except Exception:
        pass


@app.get("/health")
def health() -> Dict[str, str]:
    try:
        db.command("ping")
    except Exception:
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"status": "ok"}


class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str
    name: str
    email: str
    department: str
    shift_start: str = "09:30"
    shift_end: str = "18:30"
    joined_on: str

    @field_validator("emp_code")
    @classmethod
    def validate_emp_code(cls, v: str) -> str:
        if not v:
            raise ValueError("emp_code is required")
        return v

    @field_validator("shift_start", "shift_end")
    @classmethod
    def validate_shift_time(cls, v: str) -> str:
        if v.count(":") != 1:
            raise ValueError("shift time must be HH:MM")
        h, m = map(int, v.split(":"))
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError("invalid shift time")
        return v

    @field_validator("joined_on")
    @classmethod
    def validate_joined_on(cls, v: str) -> str:
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError("joined_on must be YYYY-MM-DD") from exc
        return v


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str
    punched_at: Optional[int] = None
    status: str = "PRESENT"

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in PRESENT_STATUSES:
            raise ValueError("status must be PRESENT, WFH or ON_DUTY")
        return v


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str
    punched_at: Optional[int] = None


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Optional[str] = None
    punch_in: Optional[int] = None
    punch_out: Optional[int] = None
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in ALL_STATUSES:
            raise ValueError("status is invalid")
        return v


def _validate_epoch_milliseconds(value: Optional[int], field: str) -> Optional[datetime]:
    if value is None:
        return None
    if not isinstance(value, int) or not (100000000000 <= value <= 4102444800000):
        raise HTTPException(status_code=422, detail=f"{field} must be a valid epoch milliseconds value")
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _attendance_date_for_punch(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    punch_in_ist = punch_in.astimezone(IST)
    if shift_end <= shift_start and punch_in_ist.time() < parse_hhmm(shift_end):
        record_date = punch_in_ist.date() - timedelta(days=1)
    else:
        record_date = punch_in_ist.date()
    return record_date.isoformat()


def compute_late_minutes(punch_in: datetime, shift_start: str, shift_end: str, attendance_date: Optional[str] = None) -> int:
    punch_in = _as_utc(punch_in)
    punch_in_ist = punch_in.astimezone(IST)
    day = date.fromisoformat(attendance_date) if attendance_date else punch_in_ist.date()
    if shift_end <= shift_start and punch_in_ist.time() < parse_hhmm(shift_end):
        day = day - timedelta(days=1)
    shift_start_time = parse_hhmm(shift_start)
    start = datetime.combine(day, shift_start_time, tzinfo=IST)
    minutes = int((punch_in_ist - start).total_seconds() / 60)
    return minutes if minutes > 10 else 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    punch_in = _as_utc(punch_in)
    punch_out = _as_utc(punch_out)
    seconds = (punch_out - punch_in).total_seconds()
    return round_half_up(seconds / 3600, 2)


def compute_overtime(punch_out: datetime, shift_start: str, shift_end: str, attendance_date: Optional[str] = None) -> int:
    punch_out = _as_utc(punch_out)
    punch_out_ist = punch_out.astimezone(IST)
    if shift_end <= shift_start:
        end_day = punch_out_ist.date()
    else:
        end_day = date.fromisoformat(attendance_date) if attendance_date else punch_out_ist.date()
    end = datetime.combine(end_day, parse_hhmm(shift_end), tzinfo=IST)
    diff = int((punch_out_ist - end).total_seconds() / 60)
    return max(0, diff) if diff >= 30 else 0


def _normalize_status_and_times(record: Dict[str, Any], status: Optional[str], punch_in: Optional[datetime], punch_out: Optional[datetime]) -> Dict[str, Any]:
    status_value = record.get("status") if status is None else status
    record_punch_in = _as_utc(record.get("punch_in")) if record.get("punch_in") is not None else None
    record_punch_out = _as_utc(record.get("punch_out")) if record.get("punch_out") is not None else None
    final_punch_in = record_punch_in if punch_in is None else _as_utc(punch_in)
    final_punch_out = record_punch_out if punch_out is None else _as_utc(punch_out)

    if status_value in {"ABSENT", "LEAVE"}:
        final_punch_in = None
        final_punch_out = None
    elif status_value in PRESENT_STATUSES:
        if final_punch_in is None:
            raise HTTPException(status_code=422, detail="presence status requires punch_in")
        if final_punch_out is not None and final_punch_out <= final_punch_in:
            raise HTTPException(status_code=422, detail="punch_out must be after punch_in")
        if final_punch_out is not None:
            elapsed_hours = (final_punch_out - final_punch_in).total_seconds() / 3600
            if elapsed_hours > 24:
                raise HTTPException(status_code=422, detail="punch_out must be within 24 hours of punch_in")

    if final_punch_in is not None and final_punch_out is not None and final_punch_out <= final_punch_in:
        raise HTTPException(status_code=422, detail="punch_out must be after punch_in")

    if final_punch_in is not None:
        punch_in_ist = final_punch_in.astimezone(IST)
        attendance_date = record.get("date")
        if attendance_date is not None:
            expected_date = date.fromisoformat(attendance_date)
            if punch_in_ist.date() != expected_date:
                raise HTTPException(status_code=422, detail="punch_in must stay on the record's attendance date")

    late_minutes = 0
    work_hours = None
    overtime_minutes = 0
    half_day = False

    if final_punch_in is not None and status_value in PRESENT_STATUSES:
        late_minutes = compute_late_minutes(final_punch_in, record["shift_start"], record["shift_end"], record.get("date"))
        if final_punch_out is not None:
            work_hours = compute_work_hours(final_punch_in, final_punch_out)
            overtime_minutes = compute_overtime(final_punch_out, record["shift_start"], record["shift_end"], record.get("date"))
            half_day = work_hours < 4.50

    return {
        "status": status_value,
        "punch_in": final_punch_in,
        "punch_out": final_punch_out,
        "late_minutes": late_minutes,
        "work_hours": work_hours,
        "overtime_minutes": overtime_minutes,
        "half_day": half_day,
    }


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    if db.employees.find_one({"emp_code": body.emp_code}):
        raise HTTPException(status_code=409, detail="emp_code already exists")
    doc = body.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    try:
        db.employees.insert_one(doc)
    except Exception:
        raise HTTPException(status_code=409, detail="emp_code already exists") from None
    return _serialize_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = {}
    if department is not None:
        q["department"] = department
    total = db.employees.count_documents(q)
    skip = (page - 1) * page_size
    items = list(db.employees.find(q, {"_id": 0}).sort("emp_code", 1).skip(skip).limit(page_size))
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(status_code=404, detail="employee not found")

    if body.punched_at is not None:
        punched_dt = _validate_epoch_milliseconds(body.punched_at, "punched_at")
    else:
        punched_dt = datetime.now(timezone.utc)

    attendance_date = _attendance_date_for_punch(punched_dt, emp["shift_start"], emp["shift_end"])
    doc = {
        "emp_code": body.emp_code,
        "date": attendance_date,
        "status": body.status,
        "punch_in": punched_dt,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": compute_late_minutes(punched_dt, emp["shift_start"], emp["shift_end"], attendance_date),
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        result = db.attendance_logs.insert_one(doc)
    except Exception:
        raise HTTPException(status_code=409, detail="already punched in for this date") from None
    created = db.attendance_logs.find_one({"_id": result.inserted_id})
    return _serialize_record(created)


@app.post("/attendance/punch-out", status_code=200)
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if emp is None:
        raise HTTPException(status_code=404, detail="employee not found")

    punched_dt = datetime.now(timezone.utc) if body.punched_at is None else _validate_epoch_milliseconds(body.punched_at, "punched_at")
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": punched_dt}, "punch_out": None},
        sort=[("punch_in", -1)],
    )
    if record is None:
        record = db.attendance_logs.find_one({"emp_code": body.emp_code}, sort=[("punch_in", -1)])
        if record is None or record.get("punch_in") is None or record.get("punch_in") > punched_dt:
            raise HTTPException(status_code=404, detail="no punch-in found")
        if record.get("punch_out") is not None:
            raise HTTPException(status_code=409, detail="that record is already punched out")

    if record.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="that record is already punched out")

    punch_in_dt = _as_utc(record["punch_in"])
    if punched_dt <= punch_in_dt:
        raise HTTPException(status_code=422, detail="punched_at must be after punch_in")
    if (punched_dt - punch_in_dt).total_seconds() > 24 * 3600:
        raise HTTPException(status_code=422, detail="punched_at is more than 24 hours after punch_in")

    final_status = record.get("status")
    if final_status not in PRESENT_STATUSES:
        raise HTTPException(status_code=422, detail="only presence statuses can be punched out")

    doc = record.copy()
    doc["punch_out"] = punched_dt
    doc["work_hours"] = compute_work_hours(_as_utc(record["punch_in"]), punched_dt)
    doc["overtime_minutes"] = compute_overtime(punched_dt, emp["shift_start"], emp["shift_end"], record["date"])
    doc["half_day"] = doc["work_hours"] < 4.50

    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {"$set": {
            "punch_out": punched_dt,
            "work_hours": doc["work_hours"],
            "overtime_minutes": doc["overtime_minutes"],
            "half_day": doc["half_day"],
        }},
        return_document=True,
    )
    if updated is None:
        raise HTTPException(status_code=409, detail="that record is already punched out")
    return _serialize_record(updated)


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must be <= date_to")

    q: Dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status:
        q["status"] = status

    total = db.attendance_logs.count_documents(q)
    skip = (page - 1) * page_size
    items = list(db.attendance_logs.find(q).sort([("date", -1), ("emp_code", 1)]).skip(skip).limit(page_size))
    return {"items": [_serialize_record(item) for item in items], "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, date: str, body: RegularizeIn):
    employee = db.employees.find_one({"emp_code": emp_code})
    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if employee is None or record is None:
        raise HTTPException(status_code=404, detail="attendance record not found")

    mutation = {} if body.status is None else {"status": body.status}
    if body.punch_in is not None:
        mutation["punch_in"] = datetime.fromtimestamp(body.punch_in / 1000, tz=timezone.utc)
    if body.punch_out is not None:
        mutation["punch_out"] = datetime.fromtimestamp(body.punch_out / 1000, tz=timezone.utc)

    if not mutation:
        raise HTTPException(status_code=422, detail="nothing to update")

    if body.status is not None and body.status not in ALL_STATUSES:
        raise HTTPException(status_code=422, detail="invalid status value")

    status_value = body.status if body.status is not None else record.get("status")
    if status_value in {"ABSENT", "LEAVE"}:
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(status_code=422, detail="ABSENT/LEAVE records cannot include punch times")
    if status_value in PRESENT_STATUSES:
        existing_punch_in = mutation.get("punch_in", record.get("punch_in"))
        if existing_punch_in is None:
            raise HTTPException(status_code=422, detail="presence status requires punch_in")
        if _as_utc(existing_punch_in).astimezone(IST).date().isoformat() != date:
            raise HTTPException(status_code=422, detail="punch_in must stay on the record's attendance date")

    final_values = {**record, **mutation}
    if body.status is not None:
        final_values["status"] = body.status
    if body.punch_in is not None:
        final_values["punch_in"] = datetime.fromtimestamp(body.punch_in / 1000, tz=timezone.utc)
    if body.punch_out is not None:
        final_values["punch_out"] = datetime.fromtimestamp(body.punch_out / 1000, tz=timezone.utc)

    if final_values["status"] in {"ABSENT", "LEAVE"}:
        final_values["punch_in"] = None
        final_values["punch_out"] = None
    elif final_values["status"] in PRESENT_STATUSES:
        if final_values.get("punch_in") is None:
            raise HTTPException(status_code=422, detail="presence status requires punch_in")
        if final_values.get("punch_out") is not None and _as_utc(final_values["punch_out"]) <= _as_utc(final_values["punch_in"]):
            raise HTTPException(status_code=422, detail="punch_out must be after punch_in")
        if final_values.get("punch_out") is not None and (_as_utc(final_values["punch_out"]) - _as_utc(final_values["punch_in"])).total_seconds() > 24 * 3600:
            raise HTTPException(status_code=422, detail="punch_out must be within 24 hours of punch_in")
        if _as_utc(final_values["punch_in"]).astimezone(IST).date().isoformat() != date:
            raise HTTPException(status_code=422, detail="punch_in must stay on the record's attendance date")

    final_data = _normalize_status_and_times(
        {**employee, "date": date, "shift_start": employee["shift_start"], "shift_end": employee["shift_end"]},
        final_values.get("status"),
        final_values.get("punch_in"),
        final_values.get("punch_out"),
    )

    changes: Dict[str, Dict[str, Any]] = {}
    for key in ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day"):
        old_value = record.get(key)
        new_value = final_data.get(key)
        if old_value != new_value:
            changes[key] = {"from": _to_epoch_ms_value(old_value), "to": _to_epoch_ms_value(new_value)}

    if not changes:
        raise HTTPException(status_code=422, detail="request changes nothing")

    final_update = {
        "status": final_data["status"],
        "punch_in": final_data["punch_in"],
        "punch_out": final_data["punch_out"],
        "work_hours": final_data["work_hours"],
        "late_minutes": final_data["late_minutes"],
        "overtime_minutes": final_data["overtime_minutes"],
        "half_day": final_data["half_day"],
    }

    history_entry = {
        "at": datetime.now(timezone.utc),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": {
            key: {"from": _to_epoch_ms_value(value["from"]), "to": _to_epoch_ms_value(value["to"])}
            for key, value in changes.items()
        },
    }

    try:
        updated = db.attendance_logs.find_one_and_update(
            {"_id": record["_id"]},
            {"$set": final_update, "$push": {"history": history_entry}},
            return_document=True,
        )
    except Exception:
        raise HTTPException(status_code=409, detail="concurrent edit detected") from None
    return _serialize_record(updated)


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$")):
    employee = db.employees.find_one({"emp_code": emp_code})
    if employee is None:
        raise HTTPException(status_code=404, detail="employee not found")

    year, mon = map(int, month.split("-"))
    start = date(year, mon, 1)
    last_day = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    end = last_day

    working_days = 0
    for day in (start + timedelta(days=n) for n in range((end - start).days + 1)):
        if day.weekday() < 5 and day >= date.fromisoformat(employee["joined_on"]):
            working_days += 1

    present_days = 0.0
    leave_days = 0
    late_count = 0
    total_late_minutes = 0
    total_overtime_minutes = 0
    for item in db.attendance_logs.find({"emp_code": emp_code, "date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}):
        if item.get("status") in PRESENT_STATUSES:
            if date.fromisoformat(item["date"]).weekday() < 5:
                present_days += 0.5 if item.get("half_day") else 1.0
        if item.get("status") == "LEAVE":
            leave_days += 1
        if item.get("late_minutes", 0) > 0:
            late_count += 1
            total_late_minutes += int(item.get("late_minutes", 0))
        total_overtime_minutes += int(item.get("overtime_minutes", 0))

    attendance_pct = None if working_days == 0 else round_half_up((present_days / working_days) * 100, 4)
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": present_days,
        "leave_days": leave_days,
        "late_count": late_count,
        "total_late_minutes": total_late_minutes,
        "total_overtime_minutes": total_overtime_minutes,
        "attendance_pct": attendance_pct,
    }


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
):
    year, mon = map(int, month.split("-"))
    start = date(year, mon, 1)
    last_day = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    end = last_day

    employee_query: Dict[str, Any] = {"joined_on": {"$lte": end.isoformat()}}
    if department is not None:
        employee_query["department"] = department
    employees = list(db.employees.find(employee_query, {"_id": 0, "emp_code": 1, "department": 1, "joined_on": 1}))
    employees_by_code = {item["emp_code"]: item for item in employees}

    headcount_by_department: Dict[str, int] = {}
    for employee in employees:
        dept = employee["department"]
        headcount_by_department[dept] = headcount_by_department.get(dept, 0) + 1

    present_days_by_department: Dict[str, float] = {}
    avg_work_hours_by_department: Dict[str, List[float]] = {}
    late_count_by_department: Dict[str, int] = {}
    total_late_minutes_by_department: Dict[str, int] = {}
    leave_count_by_department: Dict[str, int] = {}
    on_duty_count_by_department: Dict[str, int] = {}

    for item in db.attendance_logs.find({"date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}):
        emp = employees_by_code.get(item.get("emp_code"))
        if emp is None:
            continue
        dept = emp["department"]
        if item.get("status") in PRESENT_STATUSES:
            if date.fromisoformat(item["date"]).weekday() < 5:
                present_days_by_department[dept] = present_days_by_department.get(dept, 0.0) + (0.5 if item.get("half_day") else 1.0)
        if item.get("status") in PRESENT_STATUSES and item.get("work_hours") is not None:
            avg_work_hours_by_department.setdefault(dept, []).append(float(item["work_hours"]))
        if item.get("late_minutes", 0) > 0:
            late_count_by_department[dept] = late_count_by_department.get(dept, 0) + 1
            total_late_minutes_by_department[dept] = total_late_minutes_by_department.get(dept, 0) + int(item.get("late_minutes", 0))
        if item.get("status") == "LEAVE":
            leave_count_by_department[dept] = leave_count_by_department.get(dept, 0) + 1
        if item.get("status") == "ON_DUTY":
            on_duty_count_by_department[dept] = on_duty_count_by_department.get(dept, 0) + 1

    items = []
    for department_name in sorted(headcount_by_department):
        if headcount_by_department[department_name] == 0:
            continue
        work_hours = avg_work_hours_by_department.get(department_name, [])
        avg_work_hours = None if not work_hours else round_half_up(sum(work_hours) / len(work_hours), 2)
        items.append({
            "department": department_name,
            "headcount": headcount_by_department[department_name],
            "present_days": present_days_by_department.get(department_name, 0.0),
            "avg_work_hours": avg_work_hours,
            "late_count": late_count_by_department.get(department_name, 0),
            "total_late_minutes": total_late_minutes_by_department.get(department_name, 0),
            "leave_count": leave_count_by_department.get(department_name, 0),
            "on_duty_count": on_duty_count_by_department.get(department_name, 0),
        })
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = None,
):
    year, mon = map(int, month.split("-"))
    start = date(year, mon, 1)
    last_day = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    end = last_day

    employee_query = {"joined_on": {"$lte": end.isoformat()}}
    if department is not None:
        employee_query["department"] = department
    employees = {item["emp_code"]: item for item in db.employees.find(employee_query, {"_id": 0, "emp_code": 1, "name": 1, "department": 1})}

    totals: Dict[str, Dict[str, Any]] = {}
    for item in db.attendance_logs.find({"date": {"$gte": start.isoformat(), "$lte": end.isoformat()}}):
        emp_code = item.get("emp_code")
        if emp_code not in employees:
            continue
        late_minutes = int(item.get("late_minutes", 0))
        if late_minutes <= 0:
            continue
        bucket = totals.setdefault(emp_code, {"total_late_minutes": 0, "late_count": 0})
        bucket["total_late_minutes"] += late_minutes
        bucket["late_count"] += 1

    rows = []
    for emp_code, payload in totals.items():
        employee = employees[emp_code]
        rows.append({
            "emp_code": emp_code,
            "name": employee["name"],
            "department": employee["department"],
            "total_late_minutes": payload["total_late_minutes"],
            "late_count": payload["late_count"],
        })
    rows.sort(key=lambda row: (-row["total_late_minutes"], row["emp_code"]))

    ranked: List[Dict[str, Any]] = []
    current_rank = 0
    current_total = None
    for index, row in enumerate(rows, start=1):
        if current_total is None or row["total_late_minutes"] != current_total:
            current_rank = index
            current_total = row["total_late_minutes"]
        ranked.append({**row, "rank": current_rank})

    filtered = [r for r in ranked if r["rank"] <= limit]
    return {"month": month, "items": filtered}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    from_: str = Query(..., alias="from"),
    to: str = Query(...),
):
    if to < from_:
        raise HTTPException(status_code=422, detail="to must be >= from")
    if (date.fromisoformat(to) - date.fromisoformat(from_)).days + 1 > 92:
        raise HTTPException(status_code=422, detail="range must not exceed 92 days")

    employees = list(db.employees.find({"department": department}, {"_id": 0, "emp_code": 1, "joined_on": 1}))
    if not employees:
        raise HTTPException(status_code=404, detail="department not found")

    start_date = date.fromisoformat(from_)
    end_date = date.fromisoformat(to)
    day_count = (end_date - start_date).days + 1
    daily_rows = []

    records_by_day: Dict[str, List[Dict[str, Any]]] = {}
    for item in db.attendance_logs.find({"date": {"$gte": from_, "$lte": to}}):
        records_by_day.setdefault(item["date"], []).append(item)

    for offset in range(day_count):
        current = start_date + timedelta(days=offset)
        day_key = current.isoformat()
        day_records = records_by_day.get(day_key, [])
        headcount = sum(1 for emp in employees if emp["joined_on"] <= day_key)
        present_count = 0.0
        late_count = 0
        for record in day_records:
            if record.get("status") in PRESENT_STATUSES:
                present_count += 0.5 if record.get("half_day") else 1.0
            if record.get("late_minutes", 0) > 0:
                late_count += 1
        is_working_day = current.weekday() < 5
        attendance_rate = None
        if is_working_day and headcount > 0:
            attendance_rate = round_half_up(present_count / headcount, 4) if present_count else 0.0
        daily_rows.append({
            "date": day_key,
            "is_working_day": is_working_day,
            "headcount": headcount,
            "present_count": present_count,
            "late_count": late_count,
            "attendance_rate": attendance_rate,
        })

    window: List[float] = []
    for row in daily_rows:
        if row["attendance_rate"] is not None:
            window.append(float(row["attendance_rate"]))
            if len(window) > 7:
                window.pop(0)
        row["moving_avg_7d"] = None if not window else round_half_up(sum(window) / len(window), 4)
        if row["attendance_rate"] is None:
            row["moving_avg_7d"] = None

    return {"department": department, "items": daily_rows}


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: str,
    emp_code: Optional[str] = None,
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = None,
    limit: Optional[int] = Query(None, ge=1, le=50),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[str] = None,
    from_: Optional[str] = Query(None, alias="from"),
    to: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    allowed = {"attendance_list", "employee_monthly", "department_summary", "late_leaderboard", "department_trend"}
    if endpoint not in allowed:
        raise HTTPException(status_code=422, detail="invalid endpoint")

    if endpoint == "attendance_list":
        q: Dict[str, Any] = {}
        if emp_code:
            q["emp_code"] = emp_code
        if date_from or date_to:
            q["date"] = {}
            if date_from:
                q["date"]["$gte"] = date_from
            if date_to:
                q["date"]["$lte"] = date_to
        if status:
            q["status"] = status
        sort = [("date", -1), ("emp_code", 1)]
        explain = db.command(
            "explain",
            {"find": "attendance_logs", "filter": q, "sort": {"date": -1, "emp_code": 1}, "skip": (page - 1) * page_size, "limit": page_size},
            verbosity="executionStats",
        )
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    if endpoint == "employee_monthly":
        if emp_code is None or month is None:
            raise HTTPException(status_code=422, detail="emp_code and month are required")
        q = {"emp_code": emp_code, "date": {"$gte": month + "-01", "$lte": month + "-31"}}
        explain = db.command("explain", {"find": "attendance_logs", "filter": q}, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    if endpoint == "department_summary":
        if month is None:
            raise HTTPException(status_code=422, detail="month is required")
        q = {"joined_on": {"$lte": month + "-31"}} if department is None else {"department": department, "joined_on": {"$lte": month + "-31"}}
        explain = db.command("explain", {"find": "employees", "filter": q}, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "employees", "explain": explain}

    if endpoint == "late_leaderboard":
        if month is None:
            raise HTTPException(status_code=422, detail="month is required")
        q = {"date": {"$gte": month + "-01", "$lte": month + "-31"}, "late_minutes": {"$gt": 0}}
        explain = db.command("explain", {"find": "attendance_logs", "filter": q, "sort": {"late_minutes": -1, "emp_code": 1}}, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "attendance_logs", "explain": explain}

    if endpoint == "department_trend":
        if department is None or from_ is None or to is None:
            raise HTTPException(status_code=422, detail="department, from and to are required")
        q = {"department": department}
        explain = db.command("explain", {"find": "employees", "filter": q}, verbosity="executionStats")
        return {"endpoint": endpoint, "collection": "employees", "explain": explain}

    raise HTTPException(status_code=422, detail="unsupported endpoint")
