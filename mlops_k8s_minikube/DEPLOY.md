# Minikube Deploy Guide
## MLOps Stack — Phase 3

---

## Step 1 — Start Minikube

```bash
minikube start \
  --cpus=6 \
  --memory=12288 \
  --disk-size=40g \
  --driver=docker

# Verify it started
minikube status
kubectl get nodes
```

---

## Step 2 — Enable addons

```bash
# metrics-server: required for HPA (FastAPI autoscaling)
minikube addons enable metrics-server

# Verify
minikube addons list | grep metrics-server
```

---

## Step 3 — Build images INSIDE Minikube

```bash
# Point your shell's Docker CLI to Minikube's Docker daemon
# WHY: without this, images are built in your host Docker — Minikube can't see them
eval $(minikube docker-env)

# Confirm you're in Minikube's Docker context
docker info | grep Name   # should show: minikube

cd ~/mlops-learnings

# Build all three custom images
docker build -f docker/Dockerfile      -t docker-mlflow:latest  .
docker build -f docker/Dockerfile.api  -t docker-fastapi:latest .
docker build -f docker/Dockerfile.metrics -t docker-metrics:latest .

# Verify images are present inside Minikube
docker images | grep docker-
```

---

## Step 4 — Copy k8s manifests to project

```bash
cp -r ~/mlops-learnings/  # if you already have k8s/ folder skip this

# Verify folder structure
ls ~/mlops-learnings/k8s/
# 00-namespace.yaml
# 01-secret.yaml
# 02-configmap.yaml
# 03-postgres.yaml
# 04-mlflow.yaml
# 05-fastapi.yaml
# 06-metrics.yaml
# 07-monitoring.yaml
# kustomization.yaml
```

---

## Step 5 — Deploy the entire stack

```bash
cd ~/mlops-learnings

# Apply all manifests in order using kustomize
kubectl apply -k k8s/

# Expected output:
# namespace/mlops created
# namespace/monitoring created
# secret/mlops-secrets created
# configmap/mlops-config created
# configmap/prometheus-config created
# configmap/grafana-datasource created
# configmap/grafana-dashboard-provider created
# persistentvolumeclaim/postgres-pvc created
# statefulset.apps/postgres created
# service/postgres created
# ... etc
```

---

## Step 6 — Wait and watch pods start

```bash
# Watch pods in mlops namespace come up
kubectl get pods -n mlops -w

# Watch monitoring namespace
kubectl get pods -n monitoring -w

# Expected final state (may take 3-5 minutes):
# mlops namespace:
#   postgres-0     1/1   Running   0   3m
#   mlflow-xxx     1/1   Running   0   2m
#   fastapi-xxx    1/1   Running   0   1m
#   fastapi-yyy    1/1   Running   0   1m   (2 replicas)
#   metrics-xxx    1/1   Running   0   1m
#
# monitoring namespace:
#   prometheus-xxx  1/1  Running   0   2m
#   grafana-xxx     1/1  Running   0   2m
```

---

## Step 7 — Fix artifact locations (same as Docker fix)

After MLflow starts, the experiment artifact_locations in PostgreSQL will have
the wrong path from local training. Run the same fix:

```bash
# Connect to PostgreSQL pod
kubectl exec -it postgres-0 -n mlops -- \
  psql -U mlflow_user -d mlflow_db \
  -c "SELECT experiment_id, name, artifact_location FROM experiments ORDER BY experiment_id;"

# Fix artifact locations
kubectl exec -it postgres-0 -n mlops -- \
  psql -U mlflow_user -d mlflow_db -c "
UPDATE experiments
SET artifact_location = 'mlflow-artifacts:/' || experiment_id
WHERE artifact_location NOT LIKE 'mlflow-artifacts:%';
SELECT experiment_id, name, artifact_location FROM experiments ORDER BY experiment_id;
"
```

---

## Step 8 — Get service URLs

```bash
# Get Minikube IP
minikube ip
# e.g. 192.168.49.2

# Terminal 1 — MLflow
kubectl port-forward svc/mlflow -n mlops 5000:5000

# Terminal 2 — FastAPI
kubectl port-forward svc/fastapi -n mlops 8000:8000

# Terminal 3 — Prometheus
kubectl port-forward svc/prometheus -n monitoring 9090:9090

# Terminal 4 — Grafana
kubectl port-forward svc/grafana -n monitoring 3000:3000

---

## Step 9 — Run training against K8s MLflow

```bash

export MLFLOW_TRACKING_URI=http://localhost:5000

source venv/bin/activate
python src/train.py
python src/model_registry.py

---

## Step 10 — Restart FastAPI to load champion model

```bash
kubectl rollout restart deployment/fastapi -n mlops
kubectl rollout status  deployment/fastapi -n mlops

# Confirm model loaded
kubectl logs -n mlops -l app=fastapi --tail=20 | grep -E "model|loaded|ERROR"
```

---

## Step 11 — Test predictions

```bash
FASTAPI_URL=$(minikube service fastapi -n mlops --url)

# Health check
curl -s $FASTAPI_URL/health | python3 -m json.tool

# Prediction
curl -s -X POST $FASTAPI_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}' \
  | python3 -m json.tool

# Traffic loop (same as Docker phase)
python3 -c "
import requests, time, random
url = '${FASTAPI_URL}/predict'
samples = [
  {'sepal_length':5.1,'sepal_width':3.5,'petal_length':1.4,'petal_width':0.2},
  {'sepal_length':6.0,'sepal_width':2.7,'petal_length':5.1,'petal_width':1.6},
  {'sepal_length':6.9,'sepal_width':3.1,'petal_length':5.4,'petal_width':2.1},
]
count = 0
while True:
    r = requests.post(url, json=random.choice(samples))
    count += 1
    print(r.json()['class_name'], end=' ', flush=True)
    if count % 20 == 0: print(f'[{count}]')
    time.sleep(1)
"
```

---

## Quick debug commands

```bash
# Describe a stuck pod (see Events section for errors)
kubectl describe pod <pod-name> -n mlops

# View pod logs
kubectl logs -n mlops deployment/mlflow
kubectl logs -n mlops deployment/fastapi
kubectl logs -n mlops deployment/metrics
kubectl logs -n monitoring deployment/prometheus
kubectl logs -n monitoring deployment/grafana

# Follow logs live
kubectl logs -n mlops deployment/fastapi -f

# Check all resources
kubectl get all -n mlops
kubectl get all -n monitoring

# Check PVCs (storage)
kubectl get pvc -n mlops
kubectl get pvc -n monitoring

# Exec into a pod
kubectl exec -it deployment/fastapi -n mlops -- bash

# Delete and redeploy single service
kubectl delete deployment fastapi -n mlops
kubectl apply -f k8s/05-fastapi.yaml

# Nuke and rebuild everything
kubectl delete -k k8s/
kubectl apply -k k8s/
```

---

## NodePort reference

| Service    | Namespace  | NodePort | URL                             |
|------------|------------|----------|---------------------------------|
| MLflow     | mlops      | 30500    | http://$(minikube ip):30500     |
| FastAPI    | mlops      | 30800    | http://$(minikube ip):30800     |
| Prometheus | monitoring | 30090    | http://$(minikube ip):30090     |
| Grafana    | monitoring | 30300    | http://$(minikube ip):30300     |
