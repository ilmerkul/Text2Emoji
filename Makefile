mlflow_start:
	docker compose up -d
	mlflow server --backend-store-uri postgresql://user:password@localhost:5432/mlflowdb --artifacts-destination s3://bucket --host 0.0.0.0 --port 5000
