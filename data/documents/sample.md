# Acme Corp Internal Engineering Handbook

## Introduction

Acme Corp builds distributed systems for real-time logistics. Our engineering
culture is built on three pillars: reliability, observability, and developer
autonomy. Every service we ship must include health checks, structured logs,
and a runbook before it can reach production.

## On-Call Expectations

Every backend engineer rotates through on-call once every six weeks. The
on-call shift is one week long and starts on Monday at 09:00 UTC. While
on-call, engineers are expected to respond to PagerDuty alerts within fifteen
minutes during business hours and within thirty minutes outside business hours.
Acknowledging an alert is not the same as resolving it — please post a status
update in the #incidents channel within thirty minutes.

If you cannot take a shift you are scheduled for, swap directly with another
engineer and update the on-call calendar. Do not silently no-show.

## Incident Severity Levels

We use four severity levels:

- **SEV1**: Customer-facing outage or data loss. Requires immediate full-team
  response and a public status page update within ten minutes.
- **SEV2**: Significant degradation that affects more than 10% of customers.
  Page the on-call engineer; no executive escalation required by default.
- **SEV3**: Localized issue with a workaround. File a ticket; no paging.
- **SEV4**: Minor issue, deferred to normal sprint work.

## Deployment Policy

We deploy to production using blue-green deployments. All deploys must pass
the full CI suite (unit tests, integration tests, smoke tests against staging)
and be approved by at least one other engineer. No deploys on Friday
afternoons or the day before a public holiday unless explicitly approved by
an engineering director.

## Postmortem Process

Every SEV1 and SEV2 incident gets a written postmortem within seven calendar
days. The postmortem must be blameless, must include a timeline reconstructed
from logs and traces, and must list at least three concrete action items with
owners and due dates. Postmortems are stored in the engineering wiki under
`/incidents/postmortems/` and are linked from the relevant incident ticket.
