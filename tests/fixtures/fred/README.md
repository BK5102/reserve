# FRED cassette (fake values)

`cassette.json` holds FAKE responses in the FRED API's documented JSON format, written by `make_cassette.py`. No FRED API key existed when it was made, so none of the numbers are real. It lets the macro tests run with no network and no key. When a key exists, record a real cassette and keep this one as a format test.
