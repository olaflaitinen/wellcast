# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | Yes |

Security fixes are released for the latest minor version.

## Reporting a vulnerability

Please do not report security vulnerabilities through public issues.

Report them privately through
[GitHub Security Advisories](https://github.com/olaflaitinen/wellcast/security/advisories/new)
or by email to [yunus.z.imanov@helsinki.fi](mailto:yunus.z.imanov@helsinki.fi).
Include a description, the affected versions, steps to reproduce and, if
possible, a suggested fix.

You can expect an acknowledgement within 5 working days and an assessment
within 14 days. Once a fix is available, the advisory is published with
credit to the reporter unless you prefer to stay anonymous.

## Scope notes

- `Predictor.load` uses joblib, which unpickles files. Loading a model file
  from an untrusted source can execute arbitrary code; this is documented
  behavior rather than a vulnerability.
- The Streamlit app processes uploaded files in memory and does not store
  them.
