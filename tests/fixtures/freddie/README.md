# Freddie Mac layout fixtures (fake records)

Every record here is invented by hand to exercise the loader. None comes from the real dataset and none describes a real loan or person. Field positions follow Freddie Mac's official header files (`file_headers_july_2026.zip`) and File Layout (`file_layout_july_2026.xlsx`); the `legacy_*` files follow the pre-Release-47 positions, which are partly unverified (methodology M5).

- `sample_orig_2005.txt`, `sample_perf_2005.txt`: current layout (31 and 35 fields). Loan ...0001 prepays, loan ...0002 goes 90+ days late at age 6 and ends in REO disposition, loan ...0003 is 30 days late once and is still active, with one month of status XX.
- `legacy_orig_2004.txt`, `legacy_perf_2004.txt`: legacy layout (31 and 32 fields).
