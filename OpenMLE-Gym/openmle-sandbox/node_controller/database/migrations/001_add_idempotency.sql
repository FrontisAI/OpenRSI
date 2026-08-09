ALTER TABLE jobs ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(128);
ALTER TABLE jobs ADD COLUMN IF NOT EXISTS idempotency_payload_hash VARCHAR(64);

CREATE UNIQUE INDEX IF NOT EXISTS uq_jobs_api_idem
ON jobs(api_key, idempotency_key)
WHERE idempotency_key IS NOT NULL;
