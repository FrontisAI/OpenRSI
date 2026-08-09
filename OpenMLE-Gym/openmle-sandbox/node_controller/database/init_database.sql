-- init_database.sql

\c sandbox;

CREATE TABLE jobs (
    id SERIAL PRIMARY KEY,
    job_id VARCHAR(32) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    code_file_path TEXT,
    submitted_code_chars INTEGER,
    submitted_code_sha256 VARCHAR(64),
    requirements JSONB DEFAULT '[]',
    data_paths JSONB DEFAULT '[]',
    gpu_count INTEGER DEFAULT 1,
    timeout INTEGER DEFAULT 3600,
    environment JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'queued',
    progress INTEGER DEFAULT 0,
    worker_id VARCHAR(50),
    api_key VARCHAR(100) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    started_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    result JSONB,
    metrics JSONB,
    error_message TEXT,
    idempotency_key VARCHAR(128),
    idempotency_payload_hash VARCHAR(64)
);

CREATE INDEX idx_job_id ON jobs(job_id);
CREATE INDEX idx_status ON jobs(status);
CREATE INDEX idx_api_key ON jobs(api_key);
CREATE INDEX idx_created_at ON jobs(created_at);
CREATE UNIQUE INDEX uq_jobs_api_idem
    ON jobs(api_key, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE TABLE workers (
    id SERIAL PRIMARY KEY,
    worker_id VARCHAR(50) UNIQUE NOT NULL,
    docker_name VARCHAR(100),
    gpu_ids INTEGER[],
    port INTEGER,
    status VARCHAR(20) DEFAULT 'idle',
    current_job_id VARCHAR(32),
    last_heartbeat TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
