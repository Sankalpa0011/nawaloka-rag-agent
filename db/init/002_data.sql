-- ============================================================================
-- 002_data.sql - data rows, transcribed verbatim from the supplied data.sql
-- ============================================================================
-- The INSERT statements below are copied byte-for-byte from the source file:
-- no value was altered, reordered or "cleaned up". Every data quirk (the
-- literal 'Daily' day_of_week, doctors 8 and 9 having no sessions, lab_tests
-- ids starting at 9, health_packages ids starting at 5) is preserved on
-- purpose and handled downstream in 003_views_indexes.sql and the SQL tools.
-- ============================================================================

-- Specialties
INSERT INTO specialties (id, name, department) VALUES
(1, 'General Surgery & Gastroenterology', 'Surgical Sciences'),
(2, 'Neurosurgery', 'Neurosciences'),
(3, 'ENT Surgery', 'Otorhinolaryngology'),
(4, 'General Medicine', 'Internal Medicine'),
(5, 'Respiratory & Chest Medicine', 'Pulmonology'),
(6, 'Paediatrics', 'Paediatric Care'),
(7, 'Cardiology', 'Heart Centre'),
(8, 'Radiology & Imaging', 'Diagnostics'),
(9, 'Cardiac Anaesthesia & Critical Care', 'Anaesthesiology'),
(10, 'Rheumatology & Sports Medicine', 'Orthopaedics & Rehab');

-- Doctors (Using Consultants from Nawaloka Hospitals)
INSERT INTO doctors (id, name, specialty_id, qualifications, consultation_fee) VALUES
(1, 'Dr. Maiya Gunasekara', 1, 'MS, FRCS (Eng), Laparoscopic & Gastro Specialist', 3500.00),
(2, 'Dr. Punsith Gunawardena', 2, 'MS, FRCS (Neuro), Consultant Neurosurgeon', 4500.00),
(3, 'Dr. M.T.D Lakshan', 3, 'MS (ENT), DLO, Consultant ENT Surgeon', 3000.00),
(4, 'Dr. Chandima De Mel', 4, 'MD, MRCP (UK), Consultant Physician', 2800.00),
(5, 'Dr. Riaz Moujood', 5, 'MD, FCCP, Consultant Chest Specialist', 3200.00),
(6, 'Dr. Duminda Pathirana', 6, 'MD, DCH, Consultant Paediatrician', 2500.00),
(7, 'Dr. Prakash Priyadarshan', 7, 'MD, DM (Cardiology), Consultant Cardiologist', 4000.00),
(8, 'Dr. Usha Samarasinghe', 8, 'MD (Radiology), Consultant Radiologist', 2500.00),
(9, 'Dr. Sandeep K. Sharma', 9, 'MD, DA, Consultant Cardiac Intensivist', 3800.00),
(10, 'Prof. Arjuna De Silva', 4, 'MD, FRCP (Lon), Senior Consultant Physician', 4200.00),
(11, 'Dr. Harindu Wijesinghe', 10, 'MD, MRCP (UK), Consultant Rheumatologist', 3500.00);

-- Doctor Channeling Sessions
INSERT INTO channeling_sessions (id, doctor_id, day_of_week, start_time, end_time, room_number, max_patients) VALUES
-- Dr. Maiya Gunasekara
(1, 1, 'Monday', '16:00:00', '18:00:00', 'Room 102', 15),
(2, 1, 'Wednesday', '16:00:00', '18:00:00', 'Room 102', 15),

-- Dr. Punsith Gunawardena
(3, 2, 'Tuesday', '17:00:00', '19:30:00', 'Room 205', 10),
(4, 2, 'Saturday', '09:00:00', '12:00:00', 'Room 205', 15),

-- Dr. M.T.D Lakshan
(5, 3, 'Monday', '08:30:00', '11:00:00', 'Room 114', 20),
(6, 3, 'Thursday', '15:00:00', '17:30:00', 'Room 114', 20),

-- Dr. Chandima De Mel
(7, 4, 'Tuesday', '09:00:00', '12:00:00', 'Room 108', 25),
(8, 4, 'Friday', '14:00:00', '17:00:00', 'Room 108', 25),

-- Dr. Riaz Moujood
(9, 5, 'Wednesday', '10:00:00', '12:30:00', 'Room 210', 15),
(10, 5, 'Saturday', '14:00:00', '16:30:00', 'Room 210', 15),

-- Dr. Duminda Pathirana
(11, 6, 'Daily', '16:30:00', '19:00:00', 'Room 004 (Paediatric Wing)', 20),

-- Dr. Prakash Priyadarshan
(12, 7, 'Monday', '14:00:00', '17:00:00', 'Heart Centre - Room 01', 12),
(13, 7, 'Thursday', '09:00:00', '12:00:00', 'Heart Centre - Room 01', 12),

-- Prof. Arjuna De Silva
(14, 10, 'Sunday', '09:00:00', '12:00:00', 'Room 301', 18),

-- Dr. Harindu Wijesinghe
(15, 11, 'Friday', '16:00:00', '18:30:00', 'Room 112', 15);

-- Lab Tests & Prices
INSERT INTO lab_tests (id, test_code, test_name, category, price, fasting_required_hours, preparation_instructions, report_delivery_hours) VALUES
(9, 'LAB-PPBS', 'Post Prandial Blood Sugar (PPBS)', 'Biochemistry', 650.00, 0, 'Test must be taken exactly 2 hours after a standard meal.', 4),
(10, 'LAB-OGTT', 'Oral Glucose Tolerance Test (OGTT)', 'Biochemistry', 1800.00, 10, 'Requires 10-hour fasting. Multiple blood draws taken over 2 hours post 75g glucose drink.', 6),
(11, 'LAB-UFR', 'Urine Full Report (UFR)', 'Clinical Pathology', 750.00, 0, 'Collect mid-stream urine in a sterile container provided by the lab.', 3),
(12, 'LAB-SFR', 'Stool Full Report & Occult Blood', 'Clinical Pathology', 850.00, 0, 'Avoid red meat, turnips, and horseradish 48 hours prior to test.', 4),
(13, 'LAB-ELECT', 'Serum Electrolytes (Na, K, Cl)', 'Biochemistry', 2400.00, 0, 'No special preparation needed.', 6),
(14, 'LAB-DENGUE', 'Dengue NS1 Antigen & Antibody Test', 'Serology / Virology', 3200.00, 0, 'Recommended within 1-5 days of fever onset.', 3),
(15, 'LAB-VITD', 'Vitamin D3 Total (25-OH)', 'Endocrinology', 5800.00, 0, 'No fasting required. Inform lab if taking high-dose Vitamin D supplements.', 24),
(16, 'LAB-VITB12', 'Vitamin B12 Level', 'Endocrinology', 4500.00, 8, '8-hour overnight fasting recommended.', 24),
(17, 'LAB-THYROID', 'Full Thyroid Profile (FT3, FT4, TSH)', 'Endocrinology', 4800.00, 0, 'Morning sample preferred. Do not take thyroid medication before sample collection.', 12),
(18, 'LAB-PSA', 'Prostate Specific Antigen (Total PSA)', 'Tumor Markers', 3800.00, 0, 'Avoid ejaculation and vigorous exercise 48 hours prior to blood test.', 24),
(19, 'LAB-CRP', 'High Sensitivity C-Reactive Protein (hs-CRP)', 'Immunology', 2200.00, 0, 'Used to assess cardiac risk and systemic inflammation.', 6),
(20, 'LAB-TROP', 'Troponin I Quantitative (Cardiac Marker)', 'Cardiology / Emergency', 4500.00, 0, 'Emergency cardiac biomarker for acute coronary syndrome.', 2),
(21, 'LAB-CEA', 'Carcinoembryonic Antigen (CEA)', 'Tumor Markers', 4200.00, 0, 'General cancer screening marker (gastrointestinal, lung, breast).', 24),
(22, 'LAB-CA125', 'Cancer Antigen 125 (CA-125)', 'Tumor Markers', 4600.00, 0, 'Ovarian health & cancer marker screening.', 24),
(23, 'LAB-UCULT', 'Urine Culture & Antibiotic Sensitivity', 'Microbiology', 2100.00, 0, 'Clean catch mid-stream urine. Sample must be collected before starting antibiotics.', 48),
(24, 'LAB-ESR', 'Erythrocyte Sedimentation Rate (ESR)', 'Hematology', 600.00, 0, 'General marker of inflammation.', 4),
(25, 'LAB-LIPID-AD', 'Advanced Lipid & ApoB Profile', 'Cardiology', 5200.00, 12, '12-hour strict fasting. Measures LDL-C, HDL-C, Triglycerides, ApoB, and Lipoprotein(a).', 24);

-- Health Packages
INSERT INTO health_packages (id, package_name, category, price, target_audience, included_tests_and_services) VALUES
(5, 'Starter Basic Health Screening', 'Preventive Health Check', 6450.00, 'Young adults (18-30 years)', 'Full Blood Count, Fasting Blood Sugar, Urine Full Report, Body Mass Index (BMI) assessment, and Medical Officer consultation.'),
(6, 'Essential Health Check Package', 'Preventive Health Check', 10450.00, 'Adults seeking annual wellness check', 'Full Blood Count, Fasting Blood Sugar, Lipid Profile, Urine Full Report, ECG, Serum Creatinine, and General Physician Consultation.'),
(7, 'Well Woman Package (Under 40)', 'Women Health', 25150.00, 'Females under 40 years', 'FBC, FBS, Lipid Profile, Thyroid Profile (TSH), Pap Smear, Pelvic Ultrasound, Clinical Breast Examination, and Consultant Gynecologist consultation.'),
(8, 'Executive Female Screening (Above 40)', 'Women Health', 28850.00, 'Females 40 years and above', 'Full Blood Count, FBS, HbA1c, Lipid Profile, Renal Profile, Mammogram / Breast Ultrasound, Pap Smear, Bone Density Screening, and Gynecologist consultation.'),
(9, 'Classic Male Health Package (Under 40)', 'Men Health', 30300.00, 'Males under 40 years', 'Full Blood Count, FBS, Lipid Profile, Liver Function Test, Renal Function Test, ECG, Abdominal Ultrasound, and Physician Consultation.'),
(10, 'Standard Over 40 Men Package', 'Men Health', 38500.00, 'Males 40 years and above', 'FBC, FBS, HbA1c, Lipid Profile, LFT, RFT, Total PSA (Prostate), ECG, Exercise Stress Test (TMT), Chest X-Ray, and Consultant Physician Consultation.'),
(11, 'Comprehensive Cancer Screening (Men)', 'Oncology / Preventive', 64900.00, 'Men over 45 or high risk', 'CBC, Stool Occult Blood, Total PSA, CEA, AFP, Ultrasound Abdomen & Pelvis, Low-Dose CT Chest Screening, and Oncologist / Physician Review.'),
(12, 'Comprehensive Cancer Screening (Women)', 'Oncology / Preventive', 60700.00, 'Women over 40 or high risk', 'CBC, Pap Smear, CA-125, CEA, Bilateral Digital Mammogram, Breast Ultrasound, Pelvic Ultrasound, Stool Occult Blood, and Specialist Review.'),
(13, 'Pre-Employment Medical Checkup', 'Occupational Health', 8500.00, 'Job Applicants & Employees', 'Full Blood Count, Fasting Blood Sugar, Urine Full Report, Chest X-Ray (PA view), Blood Grouping & Rh, Vision Test, and Medical Fitness Certificate.'),
(14, 'Thyroid Care & Metabolic Screening', 'Endocrinology', 7800.00, 'Patients with fatigue or weight issues', 'Free T3, Free T4, TSH, Fasting Blood Sugar, Lipid Profile, Thyroid Ultrasound Scan, and Endocrinologist Review.'),
(15, 'Pre-Marital Health Screening', 'Preventive / Reproductive', 16500.00, 'Couples planning marriage', 'Full Blood Count, Blood Grouping & Rh, Thalassemia Screening (Hb Electrophoresis), Hepatitis B Surface Antigen, HIV I & II Screening, VDRL/RPR, and Genetic Counselor consultation.');

-- ---------------------------------------------------------------------------
-- Re-sync identity sequences.
-- Rows were inserted with explicit ids, which does NOT advance the underlying
-- sequence. lab_tests starts at id 9 and health_packages at id 5, so without
-- this block the next generated id would be 1 and collide on the first insert.
-- ---------------------------------------------------------------------------
SELECT setval(pg_get_serial_sequence('specialties',         'id'), (SELECT max(id) FROM specialties),         true);
SELECT setval(pg_get_serial_sequence('doctors',             'id'), (SELECT max(id) FROM doctors),             true);
SELECT setval(pg_get_serial_sequence('channeling_sessions', 'id'), (SELECT max(id) FROM channeling_sessions), true);
SELECT setval(pg_get_serial_sequence('lab_tests',           'id'), (SELECT max(id) FROM lab_tests),           true);
SELECT setval(pg_get_serial_sequence('health_packages',     'id'), (SELECT max(id) FROM health_packages),     true);
