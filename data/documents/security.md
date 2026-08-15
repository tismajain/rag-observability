# Security & Access Policy

## Account Provisioning

New engineering accounts are provisioned via the IT ticketing system within
one business day of the manager's approval. SSO is mandatory for all internal
tools. Personal accounts must never be used to access production data.

## Secret Management

All secrets live in HashiCorp Vault under team-scoped paths. Application
secrets are injected at deploy time via the platform's secret-mount adapter.
Never commit secrets to source control. If you suspect a leak, page the
security on-call immediately.

## Production Access

Production database access requires an active break-glass ticket plus
two-person approval. SSH access to production hosts is logged and audited
weekly. Quarterly access reviews remove dormant accounts and reduce blast
radius from compromised credentials.
