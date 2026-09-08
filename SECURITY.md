# Security Policy

## Supported versions

MachineLearningForecast is pre-1.0. Only the latest released version receives fixes.

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

If private vulnerability reporting is enabled on this repository, use the
**Report a vulnerability** button under the **Security** tab — that opens a
private advisory visible only to the maintainers.

If that is unavailable, open a public issue containing only "security report,
please contact me privately" and no technical details, and a maintainer will
arrange a private channel.

Please include, once a private channel is established:

- the affected version and your Julia version,
- a minimal reproducer,
- the impact you believe it has.

You can expect an acknowledgement within a week. Since MachineLearningForecast is a
modelling library with no network or authentication surface, the most likely
issues are around untrusted input handling (for example, crafted tables or
timestamps causing unbounded resource use); those are still worth reporting.
