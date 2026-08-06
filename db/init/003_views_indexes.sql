-- ============================================================================
-- 003_views_indexes.sql - where the dataset's traps are actually defused
-- ============================================================================
-- Design decision: the known data hazards are resolved HERE, in the database,
-- rather than in prompt text. A view is deterministic and testable; a prompt
-- instruction is neither. This also means the guarded free-form Text-to-SQL
-- fallback inherits the same correctness as the curated tools, because both
-- are pointed at these views.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- ---------------------------------------------------------------------------
-- 1: channeling_sessions.day_of_week is free text, and doctor 6
--         (Dr. Duminda Pathirana) has the literal value 'Daily' rather than a
--         weekday name. A naive `WHERE day_of_week = 'Tuesday'` silently drops
--         him from every weekday query.
--
-- v_sessions_expanded emits one row per (session, concrete weekday), expanding
-- 'Daily' into all seven days. is_recurring_daily is retained so the answer
-- layer can tell the user *why* the doctor matched.
-- ---------------------------------------------------------------------------
CREATE VIEW v_sessions_expanded AS
SELECT
    cs.id                     AS session_id,
    cs.doctor_id,
    d.name                    AS doctor_name,
    s.name                    AS specialty,
    s.department,
    d.consultation_fee,
    cs.day_of_week            AS day_of_week_raw,
    x.weekday,
    (lower(btrim(cs.day_of_week)) IN ('daily', 'everyday', 'every day')) AS is_recurring_daily,
    cs.start_time,
    cs.end_time,
    cs.room_number,
    cs.max_patients
FROM channeling_sessions cs
JOIN doctors     d ON d.id = cs.doctor_id
JOIN specialties s ON s.id = d.specialty_id
CROSS JOIN LATERAL unnest(
    CASE
        WHEN lower(btrim(cs.day_of_week)) IN ('daily', 'everyday', 'every day')
            THEN ARRAY['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                       'Friday', 'Saturday', 'Sunday']
        ELSE ARRAY[initcap(btrim(cs.day_of_week))]
    END
) AS x(weekday);

COMMENT ON VIEW v_sessions_expanded IS
    'One row per session per concrete weekday. Expands the literal ''Daily'' '
    'value into all seven weekdays. ALWAYS query weekday availability through '
    'this view, never through channeling_sessions.day_of_week directly.';

-- ---------------------------------------------------------------------------
-- 2: doctors 8 (Dr. Usha Samarasinghe) and 9 (Dr. Sandeep K. Sharma) have
--         no channeling_sessions rows at all. An INNER JOIN drops them from the
--         directory entirely, which reads as "this doctor does not exist".
-- 4: names carry honorifics ('Dr.', 'Prof. Arjuna De Silva'), so exact
--         matching on a surname the user typed returns nothing.
--
-- v_doctor_directory LEFT JOINs, exposes session_count so callers can state
-- "no published sessions" explicitly, and precomputes name_plain for fuzzy
-- surname matching.
-- ---------------------------------------------------------------------------
CREATE VIEW v_doctor_directory AS
SELECT
    d.id                        AS doctor_id,
    d.name                      AS doctor_name,
    regexp_replace(d.name, '^(Dr\.|Prof\.|Prof|Dr)\s+', '', 'i') AS name_plain,
    s.name                      AS specialty,
    s.department,
    d.qualifications,
    d.consultation_fee,
    COALESCE(sess.session_count, 0)          AS session_count,
    COALESCE(sess.session_count, 0) > 0      AS has_published_sessions,
    sess.available_days
FROM doctors d
LEFT JOIN specialties s ON s.id = d.specialty_id
LEFT JOIN LATERAL (
    SELECT count(*)::int                                     AS session_count,
           array_agg(DISTINCT cs.day_of_week ORDER BY cs.day_of_week) AS available_days
    FROM channeling_sessions cs
    WHERE cs.doctor_id = d.id
) sess ON TRUE;

COMMENT ON VIEW v_doctor_directory IS
    'Complete doctor directory. Uses LEFT JOIN so doctors with no channeling '
    'sessions (ids 8 and 9) are still listed, flagged by has_published_sessions '
    '= false. name_plain has the honorific stripped for fuzzy matching.';

-- ---------------------------------------------------------------------------
-- 3: health_packages.included_tests_and_services is a single unstructured
--         TEXT blob, and its item names do NOT correspond to lab_tests.test_name
--         (the blob says 'Lipid Profile'; lab_tests has 'Advanced Lipid & ApoB
--         Profile'). Joining the two tables produces confidently wrong answers.
--
-- v_package_services explodes the blob into one row per service item so
-- "which package includes a Pap Smear?" can be answered by searching within
-- the package's own text, and can return the exact matched item verbatim.
-- ---------------------------------------------------------------------------
CREATE VIEW v_package_services AS
SELECT
    hp.id            AS package_id,
    hp.package_name,
    hp.category,
    hp.price,
    hp.target_audience,
    btrim(regexp_replace(regexp_replace(item, '^\s*and\s+', '', 'i'), '\.\s*$', '')) AS service_item
FROM health_packages hp
CROSS JOIN LATERAL unnest(string_to_array(hp.included_tests_and_services, ',')) AS item
WHERE btrim(item) <> '';

COMMENT ON VIEW v_package_services IS
    'One row per service listed inside a package''s free-text '
    'included_tests_and_services blob. Item names are NOT joinable to '
    'lab_tests.test_name - search within this view instead.';

-- ---------------------------------------------------------------------------
-- Trigram indexes. These are what make fuzzy surname lookup and
-- "which package includes X" fast and typo-tolerant.
-- Expression indexes require IMMUTABLE functions; regexp_replace and lower
-- both qualify.
-- ---------------------------------------------------------------------------
CREATE INDEX idx_doctors_name_trgm
    ON doctors USING gin (name gin_trgm_ops);

CREATE INDEX idx_doctors_name_plain_trgm
    ON doctors USING gin ((regexp_replace(name, '^(Dr\.|Prof\.|Prof|Dr)\s+', '', 'i')) gin_trgm_ops);

CREATE INDEX idx_specialties_name_trgm
    ON specialties USING gin (name gin_trgm_ops);

CREATE INDEX idx_specialties_department_trgm
    ON specialties USING gin (department gin_trgm_ops);

CREATE INDEX idx_lab_tests_name_trgm
    ON lab_tests USING gin (test_name gin_trgm_ops);

CREATE INDEX idx_lab_tests_category_trgm
    ON lab_tests USING gin (category gin_trgm_ops);

CREATE INDEX idx_packages_name_trgm
    ON health_packages USING gin (package_name gin_trgm_ops);

-- The index that makes TRAP 3 tractable.
CREATE INDEX idx_packages_included_trgm
    ON health_packages USING gin (included_tests_and_services gin_trgm_ops);

CREATE INDEX idx_sessions_doctor ON channeling_sessions (doctor_id);
CREATE INDEX idx_doctors_specialty ON doctors (specialty_id);
