# Security Policy

## Supported versions

Jeeva is pre-1.0. Security fixes are provided for the latest published `0.1.x` release only.

| Version | Supported |
|---|---|
| Latest `0.1.x` | Yes |
| Older releases and unreleased snapshots | No |

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Use [GitHub private vulnerability reporting](https://github.com/waseem07799/jeeva/security/advisories/new). If that channel is unavailable, email the maintainer at `waseem07799@gmail.com` with the subject `Jeeva security report`.

Include the affected Jeeva version, operating system and Python version, a minimal reproduction, expected impact, and whether the report involves untrusted model artifacts, image inputs, authentication, or credential exposure. Remove API keys, access tokens, private model contents, and personal data from the report.

The maintainer will coordinate validation, remediation, disclosure timing, and credit through the private report. Please allow a fix to be prepared before publishing details.

## Security boundaries

Jeeva treats model repositories, saved artifacts, image files, schemas, and cloud-arbiter responses as untrusted inputs. Remote image URLs are not accepted; applications that fetch them remain responsible for network egress and peer validation. Applications are also responsible for model-source allowlists, artifact distribution, secret management, authorization around decisions, and human review for consequential outcomes.

A model decision is not a security authorization by itself. Use `is_ambiguous`, calibration evidence, and application policy as inputs to a separate authorization or review boundary.
