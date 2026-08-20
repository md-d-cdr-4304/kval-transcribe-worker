# kval-transcribe-worker

Containerized transcription worker for the [qualanalyzer](https://github.com/theobodin/qualanalyzer) project.
Chunks audio and sends it via direct-upload to a private Azure OpenAI transcription
deployment. Contains no secrets, credentials, or research data — those are all
supplied at container run time via environment variables.

## Status: temporary bridge, not the permanent home

This repo exists **only** because KI's private Azure Container Registry
(`kvalacr`) is missing its `privatelink.azurecr.io` DNS zone, which self-service
users are blocked from creating by an org-wide policy. Until KI's platform/network
team links that zone, this public GHCR image is what the Azure Container Instance
worker pulls from instead.

**Once the private DNS zone is set up**: rebuild and push this same image to
`kvalacr.azurecr.io` instead, point the ACI deployment back at it, and this repo
can be archived/deleted. Do not treat this as the long-term setup.
