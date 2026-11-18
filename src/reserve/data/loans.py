"""Loader for Freddie Mac Single-Family Loan-Level Dataset origination and performance files.

Layout source (methodology M5): Freddie Mac "Single-Family Loan-Level Dataset General
User Guide", Release 48, October 2026 (file `general_user_guide_july_2026.pdf`), the
"File Layout" spreadsheet `file_layout_july_2026.xlsx`, and the official header files
in `file_headers_july_2026.zip`, all downloaded from
https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset on 2026-11-17.

Facts taken from those documents:

- Files are pipe delimited and have no header row (User Guide, "Dataset Naming
  Convention" and "Glossary").
- Sample files are `sample_orig_YYYY.txt` and `sample_perf_YYYY.txt` inside
  `sample_YYYY.zip`; standard quarterly files are `orig_YYYYQn.txt` and `perf_YYYYQn.txt`.
- Since Release 47 (July 2026) the origination file has 31 fields and the performance
  file has 35, in the order of the header files (CURRENT_ORIG and CURRENT_PERF below).
- Files from earlier releases ("legacy") have 31 origination fields with Servicer Name
  at position 25 and no VantageScore, and 32 performance fields (the "CURRENT" column
  of Freddie Mac's "SFLLD Disclosure Changes Effective July 2026", version 1.2). The
  legacy layout is marked UNVERIFIED in methodology M5 because that PDF's text is
  partly garbled when extracted; check it against a real legacy file before use.
- Loan Identifier format is PYYQnXXXXXXX: product letter, origination year and quarter,
  then seven random digits.
- Current Loan Delinquency Status: 00 current, 01 for 30-59 days, 02 for 60-89, 03 for
  90-119, and so on, capped at 99; RA means REO Acquisition; XX means not available.
  Legacy files used unpadded digits (0, 1, 2, ...); both are accepted.
- Zero Balance Code: 01 prepaid or matured, 02 third party sale, 03 short sale or
  charge off, 09 REO disposition, 15 whole loan sales, 16 reperforming loan
  securitizations, 96 confirmed underwriting or servicing defect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass(frozen=True)
class Field:
    name: str  # snake_case name used in this project
    official: str  # attribute name in Freddie Mac's header file
    kind: str  # "str", "int", "float", or "yyyymm"


# Release 47 and later, in file order. `official` is copied from file_headers_july_2026.zip.
CURRENT_ORIG: tuple[Field, ...] = (
    Field("credit_score", "CLASSIC FICO", "int"),
    Field("first_payment_date", "FIRST PAYMENT DATE", "yyyymm"),
    Field("first_time_homebuyer", "FIRST TIME HOMEBUYER INDICATOR", "str"),
    Field("maturity_date", "MATURITY DATE", "yyyymm"),
    Field("msa", "METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION", "str"),
    Field("mi_pct", "MORTGAGE INSURANCE PERCENTAGE (MI %)", "int"),
    Field("num_units", "NUMBER OF UNITS", "int"),
    Field("occupancy_status", "OCCUPANCY STATUS", "str"),
    Field("orig_cltv", "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)", "int"),
    Field("orig_dti", "ORIGINAL DEBT-TO-INCOME (DTI) RATIO", "int"),
    Field("orig_upb", "ORIGINAL UPB", "float"),
    Field("orig_ltv", "ORIGINAL LOAN-TO-VALUE (LTV)", "int"),
    Field("orig_interest_rate", "ORIGINAL INTEREST RATE", "float"),
    Field("channel", "CHANNEL", "str"),
    Field("prepayment_penalty", "PREPAYMENT PENALTY INDICATOR", "str"),
    Field("amortization_type", "AMORTIZATION TYPE", "str"),
    Field("property_state", "PROPERTY STATE", "str"),
    Field("property_type", "PROPERTY TYPE", "str"),
    Field("postal_code", "POSTAL CODE", "str"),
    Field("loan_id", "LOAN IDENTIFIER", "str"),
    Field("loan_purpose", "LOAN PURPOSE", "str"),
    Field("orig_loan_term", "ORIGINAL LOAN TERM", "int"),
    Field("num_borrowers", "NUMBER OF BORROWERS", "int"),
    Field("seller_name", "SELLER NAME", "str"),
    Field("super_conforming", "SUPER CONFORMING FLAG", "str"),
    Field("pre_harp_loan_id", "PRE-HARP LOAN SEQUENCE NUMBER", "str"),
    Field("special_eligibility_program", "SPECIAL ELIGIBILITY PROGRAM", "str"),
    Field("harp_indicator", "HARP INDICATOR", "str"),
    Field("property_valuation_method", "PROPERTY VALUATION METHOD", "int"),
    Field("interest_only", "INTEREST ONLY (I/O) INDICATOR", "str"),
    Field("vantage_score_4", "VANTAGESCORE 4.0", "int"),
)

CURRENT_PERF: tuple[Field, ...] = (
    Field("loan_id", "LOAN IDENTIFIER", "str"),
    Field("period", "PERIOD", "yyyymm"),
    Field("current_upb", "CURRENT ACTUAL UPB", "float"),
    Field("dlq_status", "CURRENT LOAN DELINQUENCY STATUS", "str"),
    Field("loan_age", "LOAN AGE", "int"),
    Field("remaining_months", "REMAINING MONTHS TO LEGAL MATURITY", "int"),
    Field(
        "defect_settlement_date",
        "UNDERWRITING DEFECT AND MAJOR SERVICING DEFECT SETTLEMENT DATE",
        "yyyymm",
    ),
    Field("modification_flag", "MODIFICATION FLAG", "str"),
    Field("zero_balance_code", "ZERO BALANCE CODE", "int"),
    Field("zero_balance_date", "ZERO BALANCE EFFECTIVE DATE", "yyyymm"),
    Field("current_interest_rate", "CURRENT INTEREST RATE", "float"),
    Field("non_interest_bearing_upb", "CURRENT NON-INTEREST BEARING UPB", "float"),
    Field("ddlpi", "DUE DATE OF LAST PAID INSTALLMENT (DDLPI)", "yyyymm"),
    Field("mi_recoveries", "MI RECOVERIES", "float"),
    Field("net_sales_proceeds", "NET SALES PROCEEDS", "float"),
    Field("non_mi_recoveries", "NON MI RECOVERIES", "float"),
    Field("total_expenses", "TOTAL EXPENSES", "float"),
    Field("legal_costs", "LEGAL COSTS", "float"),
    Field("maintenance_costs", "MAINTENANCE AND PRESERVATION COSTS", "float"),
    Field("taxes_and_insurance", "TAXES AND INSURANCE", "float"),
    Field("misc_expenses", "MISCELLANEOUS EXPENSES", "float"),
    Field("actual_loss", "ACTUAL LOSS", "float"),
    Field("cumulative_modification_costs", "CUMULATIVE MODIFICATION COSTS", "float"),
    Field("step_modification", "INTEREST RATE STEP INDICATOR", "str"),
    Field("payment_deferral", "PAYMENT DEFERRAL FLAG", "str"),
    Field("eltv", "ESTIMATED LOAN-TO-VALUE (ELTV)", "int"),
    Field("zero_balance_removal_upb", "ZERO BALANCE REMOVAL UPB", "float"),
    Field("delinquent_accrued_interest", "DELINQUENT ACCRUED INTEREST", "float"),
    Field("disaster_delinquency", "DELINQUENCY DUE TO DISASTER", "str"),
    Field("borrower_assistance_plan", "BORROWER ASSISTANCE PLAN", "str"),
    Field("current_period_modification_costs", "CURRENT PERIOD MODIFICATION COSTS", "float"),
    Field("interest_bearing_upb", "CURRENT INTEREST BEARING UPB", "float"),
    Field("mi_cancellation", "MORTGAGE INSURANCE CANCELLATION INDICATOR", "str"),
    Field("servicer_name", "SERVICER NAME", "str"),
    Field("bankruptcy_cramdown_costs", "BANKRUPTCY CRAMDOWN COSTS", "float"),
)

# Pre-Release-47 layouts (UNVERIFIED, see the module docstring and methodology M5).
LEGACY_ORIG: tuple[Field, ...] = (
    *CURRENT_ORIG[:24],
    Field("servicer_name", "SERVICER NAME", "str"),
    *CURRENT_ORIG[24:30],
)
LEGACY_PERF: tuple[Field, ...] = CURRENT_PERF[:32]

LAYOUTS = {
    "orig": {"current": CURRENT_ORIG, "legacy": LEGACY_ORIG},
    "perf": {"current": CURRENT_PERF, "legacy": LEGACY_PERF},
}

# Codes the documents define as "not available" for numeric origination fields.
NOT_AVAILABLE = {
    "credit_score": 9999,
    "vantage_score_4": 9999,
    "mi_pct": 999,
    "orig_cltv": 999,
    "orig_dti": 999,
    "orig_ltv": 999,
    "num_units": 99,
    "num_borrowers": 99,
    "property_valuation_method": 7,
    "eltv": 999,
}

ZERO_BALANCE_CODES = {
    1: "prepaid_or_matured",
    2: "third_party_sale",
    3: "short_sale_or_charge_off",
    9: "reo_disposition",
    15: "whole_loan_sale",
    16: "reperforming_loan_securitization",
    96: "defect",
}

LOAN_ID_RE = re.compile(r"^[FA](\d{2})Q([1-4])\d{7}$")

# Performance columns the vintage builder and the data-fit check need.
PERF_CORE = (
    "loan_id",
    "period",
    "current_upb",
    "dlq_status",
    "loan_age",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_date",
    "ddlpi",
    "payment_deferral",
    "disaster_delinquency",
    "borrower_assistance_plan",
)


class LayoutError(ValueError):
    """A file does not match any known Freddie Mac layout."""


def _first_line_width(path: Path) -> tuple[int, bool]:
    """Number of pipe-separated fields on the first line, and whether a trailing pipe exists."""
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        line = f.readline().rstrip("\r\n")
    if not line:
        raise LayoutError(f"{path} is empty")
    parts = line.split("|")
    return len(parts), parts[-1] == ""


def _raw_scan(path: Path, n_fields: int) -> pl.LazyFrame:
    names = [f"f{i}" for i in range(1, n_fields + 1)]
    return pl.scan_csv(
        path,
        separator="|",
        has_header=False,
        new_columns=names,
        infer_schema=False,  # every field read as text, then cast explicitly
        quote_char=None,
        encoding="utf8-lossy",
    )


def _field_count(path: Path, allowed: set[int]) -> tuple[int, int]:
    """(fields in the layout, fields on disk). A trailing pipe adds one empty field on disk.

    The last real field is often empty too, so the exact width wins when it is allowed.
    """
    width, trailing = _first_line_width(path)
    if width in allowed:
        return width, width
    if trailing and width - 1 in allowed:
        return width - 1, width
    expected = " or ".join(str(n) for n in sorted(allowed, reverse=True))
    raise LayoutError(f"{path.name}: {width} fields on the first line; expected {expected}")


def detect_layout(path: str | Path, kind: str) -> tuple[str, int]:
    """Return ("current" or "legacy", number of fields on disk) for an orig or perf file."""
    path = Path(path)
    if kind not in LAYOUTS:
        raise ValueError("kind must be 'orig' or 'perf'")
    n, width = _field_count(path, {len(f) for f in LAYOUTS[kind].values()})
    if kind == "perf":
        return next(name for name, f in LAYOUTS["perf"].items() if len(f) == n), width
    # Both origination layouts have 31 fields. Field 31 is VantageScore (digits) in the
    # current layout and the interest-only indicator (Y or N) in the legacy one.
    last = _raw_scan(path, width).select(pl.col("f31")).head(1000).collect()["f31"].drop_nulls()
    values = set(last.str.strip_chars().to_list()) - {""}
    if values and all(v.isdigit() for v in values):
        return "current", width
    if values and values <= {"Y", "N"}:
        return "legacy", width
    if not values:
        raise LayoutError(f"{path.name}: field 31 is empty in the first rows; pass layout=")
    raise LayoutError(f"{path.name}: field 31 values {sorted(values)[:5]} match no layout")


def _cast(field: Field) -> pl.Expr:
    col = pl.col(field.name).str.strip_chars()
    col = pl.when(col == "").then(None).otherwise(col)
    if field.kind == "str":
        return col.alias(field.name)
    if field.kind == "int":
        return col.cast(pl.Int64, strict=True).alias(field.name)
    if field.kind == "float":
        return col.cast(pl.Float64, strict=True).alias(field.name)
    if field.kind == "yyyymm":
        # The first of the month stands for the whole month.
        return (col + "01").str.strptime(pl.Date, "%Y%m%d", strict=True).alias(field.name)
    raise ValueError(field.kind)


def _load(
    path: str | Path, kind: str, layout: str, columns: tuple[str, ...] | None
) -> pl.LazyFrame:
    path = Path(path)
    detected, width = detect_layout(path, kind)
    if layout == "auto":
        layout = detected
    elif layout != detected:
        raise LayoutError(f"{path.name} looks like the {detected} layout, not {layout}")
    fields = LAYOUTS[kind][layout]
    wanted = [f for f in fields if columns is None or f.name in columns]
    missing = set(columns or ()) - {f.name for f in fields}
    if missing:
        raise LayoutError(f"columns {sorted(missing)} are not in the {layout} {kind} layout")
    position = {f.name: i + 1 for i, f in enumerate(fields)}
    lf = _raw_scan(path, width).select(pl.col(f"f{position[f.name]}").alias(f.name) for f in wanted)
    return lf.with_columns(_cast(f) for f in wanted)


def read_origination(
    path: str | Path, layout: str = "auto", columns: tuple[str, ...] | None = None
) -> pl.DataFrame:
    """Read an origination file. Adds `vintage` (e.g. "2005Q1") parsed from the loan id."""
    df = _load(path, "orig", layout, columns if columns is None else (*columns, "loan_id"))
    df = df.with_columns(
        pl.when(pl.col(c) == code).then(None).otherwise(pl.col(c)).alias(c)
        for c, code in NOT_AVAILABLE.items()
        if c in df.collect_schema().names()
    ).collect()
    return df.with_columns(vintage_from_loan_id(pl.col("loan_id")).alias("vintage"))


def read_performance(
    path: str | Path, layout: str = "auto", columns: tuple[str, ...] | None = PERF_CORE
) -> pl.DataFrame:
    """Read a monthly performance file, by default only the columns the pipeline uses.

    Adds `dlq_months` (0, 1, 2, ... months delinquent; null when RA or not available)
    and `reo_acquired` (status RA).
    """
    df = _load(path, "perf", layout, columns).collect()
    return df.with_columns(parse_delinquency(pl.col("dlq_status")))


def parse_delinquency(status: pl.Expr) -> list[pl.Expr]:
    """Split the raw delinquency status into a month count and an REO flag.

    "00"/"0" -> 0, "03"/"3" -> 3, "RA" -> null with reo_acquired true, "XX" or blank -> null.
    """
    s = status.str.strip_chars()
    digits = s.str.contains(r"^\d+$")
    return [
        pl.when(digits).then(s.cast(pl.Int64, strict=False)).otherwise(None).alias("dlq_months"),
        (s == "RA").fill_null(False).alias("reo_acquired"),
    ]


def vintage_from_loan_id(loan_id: pl.Expr) -> pl.Expr:
    """Loan id "F05Q1..." gives vintage "2005Q1". Year 99 is 1999 (the dataset starts in 1999)."""
    yy = loan_id.str.slice(1, 2).cast(pl.Int64, strict=False)
    year = pl.when(yy == 99).then(1999).otherwise(2000 + yy)
    return pl.format("{}Q{}", year, loan_id.str.slice(4, 1))


def check_loan_ids(df: pl.DataFrame) -> None:
    """Raise LayoutError if any loan id does not match PYYQnXXXXXXX."""
    bad = df.filter(~pl.col("loan_id").str.contains(LOAN_ID_RE.pattern))
    if bad.height:
        raise LayoutError(f"{bad.height} loan ids do not match PYYQnXXXXXXX: {bad['loan_id'][:3]}")


def find_sample_files(raw_dir: str | Path) -> list[tuple[int, Path, Path]]:
    """(year, origination path, performance path) for every sample year present in raw_dir."""
    raw_dir = Path(raw_dir)
    out = []
    for orig in sorted(raw_dir.glob("sample_orig_*.txt")):
        year = orig.stem.removeprefix("sample_orig_")
        perf = raw_dir / f"sample_perf_{year}.txt"
        if year.isdigit() and perf.exists():
            out.append((int(year), orig, perf))
    return out


def _format_value(field: Field) -> pl.Expr:
    col = pl.col(field.name)
    if field.kind == "yyyymm":
        col = col.dt.strftime("%Y%m")
    return col.cast(pl.String).fill_null("").alias(field.name)


def write_file(df: pl.DataFrame, path: str | Path, kind: str, layout: str = "current") -> None:
    """Write a frame in the exact pipe-delimited, header-free layout (used for synthetic data).

    Columns missing from `df` are written as empty fields.
    """
    fields = LAYOUTS[kind][layout]
    out = df.with_columns(
        pl.lit(None, dtype=pl.String).alias(f.name) for f in fields if f.name not in df.columns
    )
    if "dlq_status" in out.columns and kind == "perf" and out["dlq_status"].dtype != pl.String:
        out = out.with_columns(pl.col("dlq_status").cast(pl.String).str.zfill(2))
    if kind == "perf":
        out = out.with_columns(
            pl.col("zero_balance_code").cast(pl.String).str.zfill(2).alias("zero_balance_code")
        )
    out = out.select(_format_value(f) for f in fields)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="|", include_header=False, quote_style="never")
