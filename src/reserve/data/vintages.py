"""Vintage curves: the cumulative share of each origination quarter's loans that have ever been
90+ days delinquent or defaulted, by vintage age in months.

Definitions (methodology M7 to M10):

- Vintage: the origination quarter, taken from the loan id (for example "2005Q1").
- Vintage age: months since the last month of the origination quarter. Age 1 of vintage
  2005Q1 is calendar month 2005-04. Every loan of a vintage has the same vintage age in a
  given calendar month, so each (vintage, age) cell belongs to exactly one calendar month.
  An event before age 1 (rare, since a loan cannot be 90 days late within a few months of
  its first payment) is counted at age 1, which is later than it happened, never earlier.
- Event: the first month in which the loan is 90 or more days delinquent (status 03 or
  higher), is an REO acquisition (status RA), or leaves the data through a credit-related
  zero balance code (02 third party sale, 03 short sale or charge off, 09 REO disposition,
  15 whole loan sale). Each loan has at most one event.
- Denominator: the number of loans in the vintage's origination file. Loans that prepay or
  leave the data for another reason stay in the denominator, so the curve never decreases
  and reads as "share of the original loans that ever went bad by this age".
- `at_risk` counts loans still observed and event-free at the start of an age. It is the
  denominator a hazard model needs (methodology M10).
- Cells are kept only up to the data cutoff (the last performance month), so an unobserved
  future month is never mistaken for a month with no events.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from reserve import months

DEFAULT_ZERO_BALANCE = (2, 3, 9, 15)  # credit-related terminations (methodology M7)
EVENT_DLQ_MONTHS = 3  # status 03 = 90 to 119 days delinquent


class ReconciliationError(AssertionError):
    """Counts in the vintage table do not add up to the loan-level data."""


@dataclass(frozen=True)
class VintageData:
    loans: pl.DataFrame  # one row per originated loan
    curves: pl.DataFrame  # one row per (vintage, age) cell
    cutoff: object  # last performance month used (a date)


def is_summary(perf: pl.DataFrame) -> bool:
    """True for the one-row-per-loan output of `summarize_performance`, not monthly rows."""
    return "n_perf_rows" in perf.columns


def summarize_performance(perf: pl.DataFrame, cutoff=None) -> pl.DataFrame:
    """Monthly performance rows to one row per loan (methodology M47).

    Keeps what the vintage builder needs: first and last observed month, the first event
    month, whether that event was in forbearance, the last zero balance code, and the row
    count. Rows after `cutoff` are ignored. Each Freddie Mac year can be summarized on its
    own and the summaries stacked, because a loan appears in only one year's file
    (`build` checks this), so the full monthly rows never have to sit in memory at once.
    """
    dup = perf.group_by("loan_id", "period").len().filter(pl.col("len") > 1)
    if dup.height:
        raise ReconciliationError(f"{dup.height} duplicated (loan_id, period) rows")
    if cutoff is not None:
        perf = perf.filter(pl.col("period") <= cutoff)
    perf = perf.sort("loan_id", "period")

    zb = pl.col("zero_balance_code")
    is_event = (
        (pl.col("dlq_months") >= EVENT_DLQ_MONTHS).fill_null(False)
        | pl.col("reo_acquired").fill_null(False)
        | zb.is_in(DEFAULT_ZERO_BALANCE).fill_null(False)
    )
    # Forbearance at the event month: Borrower Assistance Plan "F" or Delinquency Due to
    # Disaster "Y", whichever of the two columns the data has (methodology M20).
    forbearance = pl.lit(False)
    if "borrower_assistance_plan" in perf.columns:
        forbearance = forbearance | (pl.col("borrower_assistance_plan") == "F").fill_null(False)
    if "disaster_delinquency" in perf.columns:
        forbearance = forbearance | (pl.col("disaster_delinquency") == "Y").fill_null(False)
    perf = perf.with_columns(is_event.alias("_event"), forbearance.alias("_forb"))
    return perf.group_by("loan_id").agg(
        pl.col("period").min().alias("first_obs_period"),
        pl.col("period").max().alias("last_obs_period"),
        pl.col("period").filter(pl.col("_event")).min().alias("event_period"),
        pl.col("_forb").filter(pl.col("_event")).first().alias("event_in_forbearance"),
        zb.drop_nulls().last().alias("zero_balance_code"),
        pl.len().alias("n_perf_rows"),
    )


def _summary_and_cutoff(perf: pl.DataFrame, cutoff) -> tuple[pl.DataFrame, object]:
    """Accept monthly rows or a summary; return (summary up to cutoff, cutoff)."""
    if not is_summary(perf):
        if cutoff is None:
            cutoff = perf["period"].max()
        return summarize_performance(perf, cutoff), cutoff
    if perf["loan_id"].is_duplicated().any():
        raise ReconciliationError(
            "a loan appears in more than one performance file, so per-file summaries "
            "cannot be stacked"
        )
    last = perf["last_obs_period"].max()
    if cutoff is not None and last is not None and cutoff < last:
        raise ValueError("a summary cannot be cut earlier; summarize with that cutoff instead")
    return perf, last if cutoff is None else cutoff


def loan_outcomes(orig: pl.DataFrame, perf: pl.DataFrame, cutoff=None) -> pl.DataFrame:
    """One row per originated loan: its vintage, first and last observed month, event, and exit.

    `perf` is monthly performance rows or their `summarize_performance` output. Performance
    rows after `cutoff` are ignored. Loans with no performance rows are kept (they stay in
    the denominator and are reported by `reconcile`).
    """
    if orig["loan_id"].is_duplicated().any():
        raise ReconciliationError("duplicated loan ids in the origination data")
    per_loan, _ = _summary_and_cutoff(perf, cutoff)
    out = orig.select("loan_id", "vintage").join(per_loan, on="loan_id", how="left")
    qend = months.expr_index(months.expr_quarter_end("vintage"))

    def age(col: str) -> pl.Expr:
        raw = months.expr_index(col) - qend
        return pl.when(raw.is_null()).then(None).otherwise(pl.max_horizontal(raw, pl.lit(1)))

    exit_reason = (
        pl.when(pl.col("zero_balance_code").is_null())
        .then(pl.lit("active"))
        .when(pl.col("zero_balance_code") == 1)
        .then(pl.lit("prepaid"))
        .when(pl.col("zero_balance_code").is_in(DEFAULT_ZERO_BALANCE))
        .then(pl.lit("default"))
        .when(pl.col("zero_balance_code") == 16)
        .then(pl.lit("reperforming_sale"))
        .when(pl.col("zero_balance_code") == 96)
        .then(pl.lit("defect"))
        .otherwise(pl.lit("other"))
    )
    return out.with_columns(
        age("first_obs_period").alias("first_obs_age"),
        age("last_obs_period").alias("last_obs_age"),
        age("event_period").alias("event_age"),
        pl.when(pl.col("n_perf_rows").is_null())
        .then(pl.lit("no_performance"))
        .otherwise(exit_reason)
        .alias("exit_reason"),
        pl.col("n_perf_rows").fill_null(0),
        pl.col("event_in_forbearance").fill_null(False),
    ).sort("vintage", "loan_id")


def vintage_curves(loans: pl.DataFrame, cutoff) -> pl.DataFrame:
    """The (vintage, age) table from `loan_outcomes`, ages 1 up to the cutoff month."""
    cut = months.index(cutoff)
    n_orig = loans.group_by("vintage").agg(pl.len().alias("n_orig"))
    cells = (
        n_orig.with_columns(
            months.expr_index(months.expr_quarter_end("vintage")).alias("_qend"),
        )
        .with_columns((pl.lit(cut) - pl.col("_qend")).alias("max_age"))
        .filter(pl.col("max_age") >= 1)
        .with_columns(pl.int_ranges(1, pl.col("max_age") + 1).alias("age"))
        .explode("age")
        .with_columns(months.expr_from_index(pl.col("_qend") + pl.col("age")).alias("period"))
    )

    # A loan is at risk from its first observed age through its event age or, without an
    # event, its last observed age. A loan still in the data at the cutoff is at risk
    # through the last cell; one that vanishes earlier without a zero balance code is
    # counted in `exits_unexplained` (methodology M9).
    observed = loans.filter(pl.col("n_perf_rows") > 0).with_columns(
        pl.coalesce("event_age", "last_obs_age").alias("_end")
    )
    no_event = pl.col("event_age").is_null()
    starts = observed.group_by("vintage", pl.col("first_obs_age").alias("age")).agg(
        pl.len().alias("_start")
    )
    stops = observed.group_by("vintage", (pl.col("_end") + 1).alias("age")).agg(
        pl.len().alias("_stop")
    )
    events = (
        loans.filter(~no_event)
        .group_by("vintage", pl.col("event_age").alias("age"))
        .agg(
            pl.len().alias("new_events"),
            pl.col("event_in_forbearance").sum().alias("new_events_forbearance"),
        )
    )
    reason = pl.col("exit_reason")
    exits = (
        observed.filter(no_event)
        .group_by("vintage", pl.col("last_obs_age").alias("age"))
        .agg(
            (reason == "prepaid").sum().alias("exits_prepaid"),
            reason.is_in(["reperforming_sale", "defect", "other"]).sum().alias("exits_other"),
            ((reason == "active") & (pl.col("last_obs_period") < cutoff))
            .sum()
            .alias("exits_unexplained"),
        )
    )
    out = cells
    for part in (starts, stops, events, exits):
        out = out.join(part, on=["vintage", "age"], how="left")
    fill = ["_start", "_stop", "new_events", "new_events_forbearance"]
    fill += ["exits_prepaid", "exits_other", "exits_unexplained"]
    out = (
        out.with_columns(pl.col(fill).fill_null(0).cast(pl.Int64))
        .sort("vintage", "age")
        .with_columns(
            (pl.col("_start") - pl.col("_stop")).cum_sum().over("vintage").alias("at_risk"),
            pl.col("new_events").cum_sum().over("vintage").alias("cum_events"),
        )
        .with_columns(
            (pl.col("cum_events") / pl.col("n_orig")).alias("cum_rate"),
            (pl.col("new_events") / pl.col("n_orig")).alias("increment"),
        )
    )
    return out.select(
        "vintage",
        "age",
        "period",
        "n_orig",
        "at_risk",
        "new_events",
        "new_events_forbearance",
        "exits_prepaid",
        "exits_other",
        "exits_unexplained",
        "cum_events",
        "cum_rate",
        "increment",
    )


def build(orig: pl.DataFrame, perf: pl.DataFrame, cutoff=None) -> VintageData:
    """Loan outcomes plus vintage curves, reconciled. `cutoff` defaults to the last month.

    `perf` is monthly performance rows or their `summarize_performance` output.
    """
    per_loan, cutoff = _summary_and_cutoff(perf, cutoff)
    loans = loan_outcomes(orig, per_loan, cutoff)
    curves = vintage_curves(loans, cutoff)
    reconcile(loans, curves, orig, per_loan)
    return VintageData(loans=loans, curves=curves, cutoff=cutoff)


def reconcile(
    loans: pl.DataFrame, curves: pl.DataFrame, orig: pl.DataFrame, perf: pl.DataFrame
) -> dict[str, int]:
    """Check that every count adds up; raise ReconciliationError if not. Returns the counts.

    `perf` is monthly rows or a `summarize_performance` summary; only its loan ids are used.
    """
    counts = {
        "loans_in_origination": orig.height,
        "loans_in_outcomes": loans.height,
        "loans_in_performance": perf["loan_id"].n_unique(),
        "performance_loans_missing_from_origination": perf.select("loan_id")
        .unique()
        .join(orig.select("loan_id"), on="loan_id", how="anti")
        .height,
        "loans_without_performance": loans.filter(pl.col("n_perf_rows") == 0).height,
        "loans_with_event": loans.filter(pl.col("event_age").is_not_null()).height,
        "loans_vanishing_without_zero_balance_code": int(curves["exits_unexplained"].sum()),
    }
    if counts["loans_in_outcomes"] != counts["loans_in_origination"]:
        raise ReconciliationError("loan outcomes do not have one row per originated loan")
    if counts["performance_loans_missing_from_origination"]:
        raise ReconciliationError(
            f"{counts['performance_loans_missing_from_origination']} performance loans have no "
            "origination record"
        )
    n_by_vintage = loans.group_by("vintage").len().sort("vintage")
    n_in_curves = curves.group_by("vintage").agg(pl.col("n_orig").first()).sort("vintage")
    joined = n_in_curves.join(n_by_vintage, on="vintage", how="left")
    if (joined["n_orig"] != joined["len"]).any():
        raise ReconciliationError("n_orig differs from the loan count of a vintage")
    in_curves = set(curves["vintage"].unique().to_list())
    events_in_curves = curves["new_events"].sum()
    events_expected = loans.filter(
        pl.col("event_age").is_not_null() & pl.col("vintage").is_in(list(in_curves))
    ).height
    if events_in_curves != events_expected:
        raise ReconciliationError(
            f"curves hold {events_in_curves} events, loan outcomes {events_expected}"
        )
    last = curves.group_by("vintage").agg(pl.all().sort_by("age").last())
    if (last["cum_events"] > last["n_orig"]).any():
        raise ReconciliationError("more events than loans in a vintage")
    if curves["at_risk"].min() is not None and curves["at_risk"].min() < 0:
        raise ReconciliationError("negative at-risk count")
    if (curves.group_by("vintage").agg(pl.col("cum_rate").diff().min())["cum_rate"] < 0).any():
        raise ReconciliationError("a cumulative curve decreases")
    # at_risk(a + 1) = at_risk(a) - events(a) - exits(a) + loans first seen at a + 1.
    nxt = curves.sort("vintage", "age").with_columns(
        pl.col("at_risk").shift(-1).over("vintage").alias("_next")
    )
    leaving = pl.col("new_events") + pl.col("exits_prepaid") + pl.col("exits_other")
    leaving = leaving + pl.col("exits_unexplained")
    entering = (
        loans.filter(pl.col("n_perf_rows") > 0)
        .group_by("vintage", (pl.col("first_obs_age") - 1).alias("age"))
        .agg(pl.len().alias("_enter"))
    )
    nxt = nxt.join(entering, on=["vintage", "age"], how="left").with_columns(
        pl.col("_enter").fill_null(0)
    )
    still = pl.col("at_risk") - leaving + pl.col("_enter")
    active_at_cut = nxt.filter(pl.col("_next").is_not_null() & (pl.col("_next") != still))
    if active_at_cut.height:
        raise ReconciliationError(f"at-risk recursion fails in {active_at_cut.height} cells")
    return counts
