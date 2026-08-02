# Nawaloka Hospitals - database schema

You are writing **PostgreSQL SELECT statements** against the hospital's
operational database. All monetary values are **Sri Lankan Rupees (LKR)**.

## Hard rules

1. Emit exactly **one** `SELECT` statement. No semicolons, no DDL, no writes.
2. Only the relations listed below. Anything else is rejected before execution.
3. Prefer the **views** - they resolve known data hazards that the base tables
   do not. Reading `channeling_sessions` or `doctors` directly is usually a bug.
4. Always `LIMIT`. 200 is the hard ceiling and is applied automatically.
5. Match user-supplied text with `ILIKE '%…%'` or `similarity(col, '…') > 0.3`,
   never `=`. Names and categories in this data are not typed the way patients
   say them.

---

## Views - use these first

### `v_doctor_directory` - the complete doctor list
| column | notes |
|---|---|
| `doctor_id` | |
| `doctor_name` | includes the honorific: `Dr. …`, `Prof. …` |
| `name_plain` | honorific stripped - **match surnames against this** |
| `specialty`, `department` | |
| `qualifications` | free text |
| `consultation_fee` | numeric, LKR 2500-4500 |
| `session_count` | number of channeling sessions on record |
| `has_published_sessions` | **false for 2 of the 11 doctors** |
| `available_days` | raw `day_of_week` values, may contain `'Daily'` |

Built with a `LEFT JOIN`, so every doctor appears - including the two with no
sessions at all. Never filter this through an inner join to
`channeling_sessions`; that silently deletes them from the answer.

### `v_sessions_expanded` - one row per session **per concrete weekday**
| column | notes |
|---|---|
| `session_id`, `doctor_id`, `doctor_name` | |
| `specialty`, `department`, `consultation_fee` | |
| `day_of_week_raw` | the value as stored, e.g. `'Daily'` |
| `weekday` | **always a real weekday name - filter on this** |
| `is_recurring_daily` | true when the row came from a `'Daily'` schedule |
| `start_time`, `end_time`, `room_number`, `max_patients` | |

**This is the only correct way to answer "who is available on <day>".**

### `v_package_services` - package contents, one row per listed service
| column | notes |
|---|---|
| `package_id`, `package_name`, `category`, `price`, `target_audience` | |
| `service_item` | a single service, e.g. `Pap Smear`, `Lipid Profile` |

Use this to answer "which package includes X". **Do not join `service_item` to
`lab_tests.test_name`** - see hazard 3.

---

## Base tables

`specialties(id, name, department)` - 10 rows
`doctors(id, name, specialty_id → specialties.id, qualifications, consultation_fee)` - 11 rows
`channeling_sessions(id, doctor_id → doctors.id, day_of_week, start_time, end_time, room_number, max_patients)` -15 rows
`lab_tests(id, test_code UNIQUE, test_name, category, price, fasting_required_hours, preparation_instructions, report_delivery_hours)` - 17 rows
`health_packages(id, package_name, category, price, target_audience, included_tests_and_services)` - 11 rows

---

## Known data hazards

**1. `day_of_week` is free text and one row says `'Daily'`.**
Dr. Duminda Pathirana (Paediatrics) is stored as `'Daily'`, not as seven weekday
rows. `WHERE day_of_week = 'Tuesday'` omits him from every weekday query.
→ Filter `v_sessions_expanded.weekday`.

**2. Two doctors have no `channeling_sessions` rows at all.**
Dr. Usha Samarasinghe (Radiology & Imaging) and Dr. Sandeep K. Sharma (Cardiac
Anaesthesia & Critical Care). They work at the hospital but publish no
channeling times. An inner join drops them entirely - the bot then implies they
do not exist. Equally, never invent times for them.
→ Use `v_doctor_directory` and report `has_published_sessions`.

**3. Package contents do not correspond to `lab_tests`.**
`included_tests_and_services` is one free-text blob. Its wording differs from the
test catalogue: the package says `Lipid Profile`, `FBC`, `FBS`; `lab_tests` has
`Advanced Lipid & ApoB Profile`. **Joining the two returns zero rows.**
→ Search inside the text (`ILIKE`) or use `v_package_services`.

**4. Doctor names carry honorifics.**
Stored as `Dr. Maiya Gunasekara`, `Prof. Arjuna De Silva`. A patient types
"De Silva". Exact match returns nothing.
→ `similarity(name_plain, :q) > 0.3` or `name_plain ILIKE '%…%'`.

**5. The website advertises packages the database does not contain.**
The public site lists e.g. "Premier Under 40 Men" (LKR 45,000) which has no row
here. **The database is the system of record.** Answer package and price
questions from these tables only.

---

## Actual column values

`channeling_sessions.day_of_week` - `Daily`, `Friday`, `Monday`, `Saturday`,
`Sunday`, `Thursday`, `Tuesday`, `Wednesday`

`specialties` (name → department):
General Surgery & Gastroenterology → Surgical Sciences ·
Neurosurgery → Neurosciences ·
ENT Surgery → Otorhinolaryngology ·
General Medicine → Internal Medicine ·
Respiratory & Chest Medicine → Pulmonology ·
Paediatrics → Paediatric Care ·
Cardiology → Heart Centre ·
Radiology & Imaging → Diagnostics ·
Cardiac Anaesthesia & Critical Care → Anaesthesiology ·
Rheumatology & Sports Medicine → Orthopaedics & Rehab

`lab_tests.category` - Biochemistry, Cardiology, Cardiology / Emergency,
Clinical Pathology, Endocrinology, Hematology, Immunology, Microbiology,
Serology / Virology, Tumor Markers

`health_packages.category` - Endocrinology, Men Health, Occupational Health,
Oncology / Preventive, Preventive / Reproductive, Preventive Health Check,
Women Health

`health_packages.target_audience` - Adults seeking annual wellness check ·
Couples planning marriage · Females 40 years and above · Females under 40 years ·
Job Applicants & Employees · Males 40 years and above · Males under 40 years ·
Men over 45 or high risk · Patients with fatigue or weight issues ·
Women over 40 or high risk · Young adults (18-30 years)

`lab_tests.fasting_required_hours` - 0, 8, 10, 12

Ranges: consultation fees 2,500–4,500 · lab tests 600–5,800 ·
packages 6,450–64,900 (all LKR).

---

## Worked examples

**"Which doctors are available on Tuesday?"** - catches the `'Daily'` row
```sql
SELECT DISTINCT doctor_name, specialty, start_time, end_time, room_number,
       is_recurring_daily
FROM v_sessions_expanded
WHERE weekday = 'Tuesday'
ORDER BY start_time
LIMIT 50;
```

**"List all doctors with their consultation fees."** - keeps the two with no sessions
```sql
SELECT doctor_name, specialty, consultation_fee, has_published_sessions
FROM v_doctor_directory
ORDER BY consultation_fee
LIMIT 50;
```

**"Which package includes a Pap Smear?"** - never joins to `lab_tests`
```sql
SELECT DISTINCT package_name, price, target_audience, service_item
FROM v_package_services
WHERE service_item ILIKE '%pap smear%'
ORDER BY price
LIMIT 20;
```

**"What is Prof. De Silva's fee?"** - fuzzy match, honorific stripped
```sql
SELECT doctor_name, specialty, consultation_fee
FROM v_doctor_directory
WHERE similarity(name_plain, 'De Silva') > 0.3
ORDER BY similarity(name_plain, 'De Silva') DESC
LIMIT 5;
```

**"Which tests require fasting, and how should I prepare?"**
Only three tests require fasting (8h, 10h and 12h). Quote
`preparation_instructions` verbatim - never paraphrase a fasting instruction.
```sql
SELECT test_name, price, fasting_required_hours, preparation_instructions
FROM lab_tests
WHERE fasting_required_hours > 0
ORDER BY fasting_required_hours DESC
LIMIT 20;
```

**"Cheapest cardiologist and when they channel"**
```sql
SELECT d.doctor_name, d.consultation_fee, s.weekday, s.start_time, s.end_time
FROM v_doctor_directory d
LEFT JOIN v_sessions_expanded s ON s.doctor_id = d.doctor_id
WHERE d.specialty ILIKE '%cardiolog%'
ORDER BY d.consultation_fee, s.weekday
LIMIT 20;
```
