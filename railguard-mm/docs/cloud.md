# Cloud deployment design

The cloud side is containerized and provider-neutral.

## Services

- **Mosquitto** — MQTT QoS-1 broker with persistent storage for the local reference stack.
- **Ingestor** — validates telemetry and writes TimescaleDB.
- **TimescaleDB** — measured and predicted time-series values.
- **MinIO** — event imagery/raw windows/model artifacts.
- **ML worker** — periodic inference over recent windows.
- **FastAPI** — read-only dashboard API and device summaries.
- **React/Nginx** — static web application.
- **Caddy** — optional public reverse proxy and TLS termination.

## Public deployment pattern

```text
Internet
   |
HTTPS :443
   |
 Caddy
   |-------------------> React static app
   |
   +-------------------> /api/* -> FastAPI

Private container network:
FastAPI -> TimescaleDB
Ingestor -> TimescaleDB
ML worker -> TimescaleDB
Edge SQLite outbox -> MQTT QoS1 broker (persistent subscriber session)
Ingestor -> TimescaleDB commit -> device-specific application ACK -> Edge outbox delete
Edge -> authenticated event upload -> FastAPI -> MinIO
```

The default Docker Compose file binds Postgres, MQTT, MinIO and FastAPI to loopback for local development. Mosquitto persistence and a stable non-clean ingestor session preserve QoS-1 messages while the ingestor is temporarily offline. The edge additionally retains each SQLite outbox row after broker PUBACK and deletes it only after a device-specific application ACK emitted after the TimescaleDB write path returns successfully. The ACK identity is exactly `(device_id, canonical-UTC timestamp, sequence)`, which is also the telemetry database idempotency key; prediction rows preserve the source sequence. The reference configuration HMAC-SHA256 signs ACKs so a publisher without the shared key cannot simply forge deletion messages. If an ACK is lost, the edge retries the same immutable identity and the cloud upsert remains idempotent. This provides at-least-once application delivery across the reference path; it is deliberately **not** described as globally exactly-once. For an internet deployment, replace the development HMAC key, terminate HTTPS at Caddy (or an equivalent reverse proxy), configure broker TLS/authentication and ACLs separately, set a write API key for artifact upload, and keep Postgres/MinIO administration private.

## API examples

```text
GET /v1/devices
GET /v1/series/{device_id}?minutes=30
GET /v1/latest/{device_id}
GET /health
```

The `series` response returns measured points and predictions keyed by target timestamp, which lets the dashboard plot them on the same x-axis without confusing prediction issue time with target time.
