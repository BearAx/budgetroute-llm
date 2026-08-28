# Kubernetes reference deployment

This manifest demonstrates three independent BudgetRoute API replicas using the PostgreSQL operational store. It is a topology reference, not a ready-made availability or capacity claim.

Before applying it:

1. Build and scan an image from an immutable source commit, then replace `REPLACE_WITH_IMMUTABLE_TAG`.
2. Create the `budgetroute` namespace and a `budgetroute-secrets` Secret through the cluster's external-secret integration. It must contain `postgres-migration-dsn`, `postgres-dsn`, and `tenant-keys-json`; do not generate a plaintext Secret manifest in this repository.
3. Give the migration DSN schema DDL rights and the runtime DSN only the documented DML/sequence rights. Both must use `sslmode=verify-full` (preferred), `verify-ca`, or `require`. Configure CA material according to the platform rather than disabling certificate validation silently.
4. Adjust the ingress `NetworkPolicy` for the actual ingress-controller namespace, add an explicit egress policy for DNS, the model service, telemetry, and PostgreSQL, and keep every other route denied.
5. Put an authenticated HTTPS ingress/WAF in front of the ClusterIP Service. The application profile asserts external TLS termination and must not be exposed directly.
6. Replace CPU/memory requests and HPA thresholds with values derived from target-runtime load tests. GPU/model servers normally run as separate private deployments.

Create the namespace before applying the remaining manifest:

```bash
kubectl create namespace budgetroute
kubectl apply -f deploy/kubernetes/migrate-job.yaml
kubectl -n budgetroute wait --for=condition=complete job/budgetroute-migrate --timeout=120s
kubectl apply -f deploy/kubernetes/budgetroute.yaml
kubectl -n budgetroute rollout status deployment/budgetroute
```

Delete or archive the completed migration Job after preserving its status/log metadata. The Deployment never receives the DDL credential; only the separate Job references `postgres-migration-dsn`.

The `emptyDir` state is intentional: API replicas do not own durable operational data. Confidence adaptation stays an operator-controlled release workflow; publish a reviewed immutable calibration artifact with the next image/config release instead of mutating different pod files independently.
