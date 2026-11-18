"""Freddie Mac loader on hand-built fake records in the exact file layout."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from reserve.data import loans

FIX = Path(__file__).parent / "fixtures" / "freddie"

# Copied from Freddie Mac's official header files (file_headers_july_2026.zip, 2026-11-17).
OFFICIAL_ORIG_HEADER = (
    "CLASSIC FICO|FIRST PAYMENT DATE|FIRST TIME HOMEBUYER INDICATOR|MATURITY DATE|"
    "METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION|"
    "MORTGAGE INSURANCE PERCENTAGE (MI %)|NUMBER OF UNITS|OCCUPANCY STATUS|"
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)|ORIGINAL DEBT-TO-INCOME (DTI) RATIO|ORIGINAL UPB|"
    "ORIGINAL LOAN-TO-VALUE (LTV)|ORIGINAL INTEREST RATE|CHANNEL|PREPAYMENT PENALTY INDICATOR|"
    "AMORTIZATION TYPE|PROPERTY STATE|PROPERTY TYPE|POSTAL CODE|LOAN IDENTIFIER|LOAN PURPOSE|"
    "ORIGINAL LOAN TERM|NUMBER OF BORROWERS|SELLER NAME|SUPER CONFORMING FLAG|"
    "PRE-HARP LOAN SEQUENCE NUMBER|SPECIAL ELIGIBILITY PROGRAM|HARP INDICATOR|"
    "PROPERTY VALUATION METHOD|INTEREST ONLY (I/O) INDICATOR|VANTAGESCORE 4.0"
)
OFFICIAL_PERF_HEADER = (
    "LOAN IDENTIFIER|PERIOD|CURRENT ACTUAL UPB|CURRENT LOAN DELINQUENCY STATUS|LOAN AGE|"
    "REMAINING MONTHS TO LEGAL MATURITY|"
    "UNDERWRITING DEFECT AND MAJOR SERVICING DEFECT SETTLEMENT DATE|MODIFICATION FLAG|"
    "ZERO BALANCE CODE|ZERO BALANCE EFFECTIVE DATE|CURRENT INTEREST RATE|"
    "CURRENT NON-INTEREST BEARING UPB|DUE DATE OF LAST PAID INSTALLMENT (DDLPI)|MI RECOVERIES|"
    "NET SALES PROCEEDS|NON MI RECOVERIES|TOTAL EXPENSES|LEGAL COSTS|"
    "MAINTENANCE AND PRESERVATION COSTS|TAXES AND INSURANCE|MISCELLANEOUS EXPENSES|ACTUAL LOSS|"
    "CUMULATIVE MODIFICATION COSTS|INTEREST RATE STEP INDICATOR|PAYMENT DEFERRAL FLAG|"
    "ESTIMATED LOAN-TO-VALUE (ELTV)|ZERO BALANCE REMOVAL UPB|DELINQUENT ACCRUED INTEREST|"
    "DELINQUENCY DUE TO DISASTER|BORROWER ASSISTANCE PLAN|CURRENT PERIOD MODIFICATION COSTS|"
    "CURRENT INTEREST BEARING UPB|MORTGAGE INSURANCE CANCELLATION INDICATOR|SERVICER NAME|"
    "BANKRUPTCY CRAMDOWN COSTS"
)


def test_field_order_matches_official_headers():
    assert "|".join(f.official for f in loans.CURRENT_ORIG) == OFFICIAL_ORIG_HEADER
    assert "|".join(f.official for f in loans.CURRENT_PERF) == OFFICIAL_PERF_HEADER
    assert len(loans.CURRENT_ORIG) == 31 and len(loans.CURRENT_PERF) == 35
    assert len(loans.LEGACY_ORIG) == 31 and len(loans.LEGACY_PERF) == 32


def test_layout_detection():
    assert loans.detect_layout(FIX / "sample_orig_2005.txt", "orig")[0] == "current"
    assert loans.detect_layout(FIX / "sample_perf_2005.txt", "perf")[0] == "current"
    assert loans.detect_layout(FIX / "legacy_orig_2004.txt", "orig")[0] == "legacy"
    assert loans.detect_layout(FIX / "legacy_perf_2004.txt", "perf")[0] == "legacy"
    with pytest.raises(loans.LayoutError):
        loans.read_origination(FIX / "sample_orig_2005.txt", layout="legacy")


def test_origination_values_and_not_available_codes():
    df = loans.read_origination(FIX / "sample_orig_2005.txt")
    assert df.height == 3
    first = df.row(0, named=True)
    assert first["loan_id"] == "F05Q10000001"
    assert first["credit_score"] == 740
    assert first["first_payment_date"] == date(2005, 3, 1)
    assert first["orig_upb"] == 200000.0
    assert first["orig_interest_rate"] == pytest.approx(5.75)
    assert first["vintage"] == "2005Q1"
    assert first["vantage_score_4"] is None  # 9999 = not available
    assert first["property_valuation_method"] is None  # 7 = not available
    assert df["credit_score"][1] is None and df["vantage_score_4"][1] == 712
    assert df["orig_ltv"][2] is None  # 999 = not available
    assert df["vintage"].to_list() == ["2005Q1", "2005Q2", "2005Q2"]
    loans.check_loan_ids(df)


def test_performance_status_codes():
    df = loans.read_performance(FIX / "sample_perf_2005.txt")
    assert df.height == 6 + 10 + 6
    assert set(loans.PERF_CORE) <= set(df.columns)
    two = df.filter(pl.col("loan_id") == "F05Q20000002").sort("period")
    assert two["dlq_months"].to_list() == [0, 0, 0, 1, 2, 3, 4, 5, None, None]
    assert two["reo_acquired"].to_list() == [False] * 8 + [True, True]
    assert two["zero_balance_code"].to_list()[-1] == 9
    assert two["zero_balance_date"].to_list()[-1] == date(2006, 3, 1)
    three = df.filter(pl.col("loan_id") == "F05Q20000003").sort("period")
    assert three["dlq_months"].to_list() == [0, 0, 1, 0, None, 0]  # "XX" -> not available
    assert not three["reo_acquired"].any()
    one = df.filter(pl.col("loan_id") == "F05Q10000001")
    assert one["zero_balance_code"].drop_nulls().to_list() == [1]


def test_full_performance_columns_and_loss_fields():
    df = loans.read_performance(FIX / "sample_perf_2005.txt", columns=None)
    assert df.columns[:35] == [f.name for f in loans.CURRENT_PERF]
    last = df.filter(pl.col("zero_balance_code") == 9).row(0, named=True)
    assert last["net_sales_proceeds"] == -110000.0  # recoveries are negative (Release 47 on)
    assert last["actual_loss"] == 42000.0
    assert last["zero_balance_removal_upb"] == 149000.0


def test_legacy_files_load_with_unpadded_status():
    orig = loans.read_origination(FIX / "legacy_orig_2004.txt")
    assert "vantage_score_4" not in orig.columns
    assert orig["servicer_name"][0] == "OTHER" and orig["interest_only"][0] == "N"
    assert orig["vintage"][0] == "2004Q1"
    perf = loans.read_performance(FIX / "legacy_perf_2004.txt")
    assert perf["dlq_months"].to_list() == [0, 1, 0]


def test_trailing_pipe_is_tolerated(tmp_path):
    src = (FIX / "sample_orig_2005.txt").read_text().splitlines()
    path = tmp_path / "sample_orig_2005.txt"
    path.write_text("\n".join(line + "|" for line in src) + "\n")
    df = loans.read_origination(path)
    assert df.height == 3 and df["loan_id"][0] == "F05Q10000001"


def test_wrong_width_is_rejected(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("a|b|c\n")
    with pytest.raises(loans.LayoutError):
        loans.read_performance(path)


def test_vintage_from_loan_id_century():
    s = pl.DataFrame({"loan_id": ["F99Q40000001", "F00Q10000001", "F26Q20000001"]})
    out = s.select(loans.vintage_from_loan_id(pl.col("loan_id")).alias("v"))["v"].to_list()
    assert out == ["1999Q4", "2000Q1", "2026Q2"]


def test_writer_round_trips(tmp_path):
    orig = loans.read_origination(FIX / "sample_orig_2005.txt", columns=None)
    perf = loans.read_performance(FIX / "sample_perf_2005.txt", columns=None)
    loans.write_file(orig.drop("vintage"), tmp_path / "sample_orig_2005.txt", "orig")
    perf_raw = perf.drop("dlq_months", "reo_acquired")
    loans.write_file(perf_raw, tmp_path / "sample_perf_2005.txt", "perf")
    orig2 = loans.read_origination(tmp_path / "sample_orig_2005.txt", columns=None)
    perf2 = loans.read_performance(tmp_path / "sample_perf_2005.txt", columns=None)
    # Not-available codes become null on read, so compare the parts that survive a round trip.
    assert orig2.select("loan_id", "first_payment_date", "orig_upb").equals(
        orig.select("loan_id", "first_payment_date", "orig_upb")
    )
    assert perf2.select("loan_id", "period", "dlq_status", "zero_balance_code").equals(
        perf.select("loan_id", "period", "dlq_status", "zero_balance_code")
    )


def test_find_sample_files(tmp_path):
    for name in ("sample_orig_2005.txt", "sample_perf_2005.txt", "sample_orig_2006.txt"):
        (tmp_path / name).write_text("x\n")
    found = loans.find_sample_files(tmp_path)
    assert [(y, o.name, p.name) for y, o, p in found] == [
        (2005, "sample_orig_2005.txt", "sample_perf_2005.txt")
    ]
