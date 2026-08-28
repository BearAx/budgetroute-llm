CREATE TABLE quota_windows (
    tenant_id TEXT NOT NULL,
    window_start DOUBLE PRECISION NOT NULL,
    request_count BIGINT NOT NULL CHECK (request_count >= 0),
    PRIMARY KEY (tenant_id, window_start)
);

CREATE TABLE admission_leases (
    lease_id UUID PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    replica_id TEXT NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    expires_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX idx_admission_leases_expires ON admission_leases(expires_at);

CREATE TABLE predictions (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    route TEXT NOT NULL,
    backend_confidence_raw DOUBLE PRECISION,
    backend_confidence DOUBLE PRECISION,
    features_json JSONB NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant_id, request_id)
);
CREATE INDEX idx_predictions_created_at ON predictions(created_at DESC);

CREATE TABLE feedback (
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    correct BOOLEAN NOT NULL,
    created_at DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (tenant_id, request_id)
);
CREATE INDEX idx_feedback_created_at ON feedback(created_at);

CREATE TABLE review_cases (
    sequence BIGSERIAL PRIMARY KEY,
    case_id UUID NOT NULL UNIQUE,
    tenant_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('open', 'claimed', 'resolved', 'dismissed')),
    assignee TEXT,
    outcome TEXT CHECK (
        outcome IS NULL OR outcome IN ('approved', 'corrected', 'rejected', 'not_actionable')
    ),
    correct BOOLEAN,
    version BIGINT NOT NULL CHECK (version >= 1),
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    UNIQUE (tenant_id, request_id)
);
CREATE INDEX idx_review_cases_tenant_status
    ON review_cases(tenant_id, status, sequence);

CREATE TABLE metrics (
    name TEXT PRIMARY KEY,
    value DOUBLE PRECISION NOT NULL
);

CREATE TABLE audit_events (
    sequence BIGSERIAL PRIMARY KEY,
    event_id UUID NOT NULL UNIQUE,
    occurred_at DOUBLE PRECISION NOT NULL,
    tenant_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    details_json JSONB NOT NULL,
    previous_hash CHAR(64) NOT NULL,
    event_hash CHAR(64) NOT NULL
);
CREATE INDEX idx_audit_events_tenant_sequence ON audit_events(tenant_id, sequence);
