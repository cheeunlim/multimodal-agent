#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

if [ -n "$1" ]; then
    PROJECT_ID="$1"
fi

if [ -z "$PROJECT_ID" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
fi

# Check if the PROJECT_ID environment variable is set
if [ -z "$PROJECT_ID" ]; then
    echo "Error: PROJECT_ID environment variable is not set."
    echo "Please run 'export PROJECT_ID=your-project-id' before executing this script."
    exit 1
fi

echo "Using Project ID: ${PROJECT_ID}"

echo "0. Enabling required GCP APIs (idempotent, may take ~1 min on a fresh project)..."
gcloud services enable \
    vectorsearch.googleapis.com \
    aiplatform.googleapis.com \
    discoveryengine.googleapis.com \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    storage.googleapis.com \
    --project="${PROJECT_ID}"

echo "1. Installing and upgrading required Python packages..."
pip install --quiet --upgrade google-cloud-vectorsearch fsspec pandas gcsfs google-auth google-api-core google-genai google-cloud-aiplatform google-cloud-discoveryengine Pillow opencv-python numpy scikit-learn seaborn ipywidgets pyOpenSSL qrcode

echo "2. Creating GCS bucket (Location: asia-northeast1)..."
gcloud storage buckets create gs://${PROJECT_ID}-vs2 --location=asia-northeast1 || true

echo "3. Copying dataset to the created GCS bucket..."
if ! gcloud storage cp gs://jk-amazon-products-index/compact-records/amazon-product-dataset-768-compact.jsonl gs://${PROJECT_ID}-vs2/data/; then
    echo ""
    echo "Error: could not read gs://jk-amazon-products-index/ (the shared dataset bucket)."
    echo "Ask the instructor to confirm the bucket is readable from this project, then re-run."
    exit 1
fi

echo "4. Starting the index builder in the background..."
nohup python3 session2_index_builder.py > index_builder.log 2>&1 &
echo "   PID $!  |  Progress: tail -f index_builder.log"

echo ""
echo "Setup done. NOTE: the index builder is STILL RUNNING in the background."
echo "It creates the collection, imports ~99k products, and builds 2 ScaNN indexes"
echo "(roughly 20-40 minutes in total). Part 2 can start before it finishes --"
echo "searches work without indexes, just a bit slower."
echo ""
echo "  Check progress : tail -f index_builder.log"
echo "  Check on GCP   : gcloud vector-search operations list --location=asia-northeast1"