# ==========================
# Vibe+ – Makefile
# ==========================

# Default Python command (change to python3 if needed)
PYTHON := python

# ==========================
#  Utilisation
# ==========================
# Prepare the data:
#   make prepare
#
# Build the features:
#   make features
#
# Train the models:
#   make train
#
# Evaluate the models:
#   make evaluate
#
# Run the full pipeline:
#   make pipeline
#
# Run everything:
#   make all
#
# Clean Python cache:
#   make clean

# ==========================
#  Python / Project Actions
# ==========================

# Prepare the data
prepare:
	$(PYTHON) -m src.01_prepare.make_dataset

# Build the features
features:
	$(PYTHON) -m src.02_features.build_features

# Train models
train:
	$(PYTHON) -m src.03_models.train_model

# Evaluate models
evaluate:
	$(PYTHON) -m src.04_evaluation.evaluate_model

# Run the entire project pipeline
pipeline:
	$(PYTHON) -m src.pipeline.main

# Clean Python cache and temp files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run everything in sequence
all: prepare features train evaluate pipeline

# ==========================
#  Package Actions
# ==========================
reinstall_package:
	@pip uninstall -y vibe || :
	@pip install -e .

# ==========================
#  Fast API
# ==========================
# Test API locally
fast_api:
	uvicorn vibe_api:app --reload

# Find it in the browser at : http://localhost:8000/docs or http://localhost:8000/predict

# ==========================
#  Docker Local
# ==========================
build_container_local:
	docker build --tag=$$IMAGE:dev .

run_container_local:
	docker run -it -e PORT=8000 -p 8080:8000 $$IMAGE:dev

# ==========================
#  Docker Deployed / Cloud Run
# ==========================

# Step 1 (ONLY FIRST TIME)
allow_docker_push:
	gcloud auth configure-docker $$GCP_REGION-docker.pkg.dev

# Step 2 (ONLY FIRST TIME)
create_artifacts_repo:
	gcloud artifacts repositories create $$ARTIFACTSREPO --repository-format=docker \
	--location=$$GCP_REGION --description="Repository for storing images"

# Step 3 (Windows or non-M Mac chips)
build_for_production:
	docker build -t $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod .

# Step 3 (⚠️ M1/M2/M3/M4 Mac)
m1_build_image_production:
	docker build --platform linux/amd64 -t $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod .

# Step 4
push_image_production:
	docker push $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod

# Step 5
deploy_to_cloud_run:
	gcloud run deploy --image $$GCP_REGION-docker.pkg.dev/$$GCP_PROJECT/$$ARTIFACTSREPO/$$IMAGE:prod --memory $$MEMORY --region $$GCP_REGION

# Disabling the Service
cloud_run_disable_service:
	gcloud run services update $$INSTANCE --min-instances=0

# Delete the Service
cloud_run_delete_service:
	gcloud run services delete $$INSTANCE
