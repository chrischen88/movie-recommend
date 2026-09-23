PY      := backend/.venv/bin/python
PIP     := backend/.venv/bin/pip
EXPORT  ?= sample_letterboxd_export.zip

.PHONY: setup setup-backend setup-frontend ingest sample build-index train serve serve-api serve-web test

setup: setup-backend setup-frontend

setup-backend:
	python3 -m venv backend/.venv
	$(PIP) install --upgrade pip
	$(PIP) install -e "backend[dev]"
	@test -f .env || (cp .env.example .env && echo "created .env — add your API keys")

setup-frontend:
	cd frontend && npm install

# Write the synthetic 30-film test export to ./sample_letterboxd_export.zip
sample:
	cd backend && .venv/bin/python -m tests.fixtures.sample_export ../$(EXPORT)

# make ingest EXPORT=~/Downloads/letterboxd-you-2026-09-01-utc.zip
ingest:
	cd backend && .venv/bin/python -m app.cli ingest $(abspath $(EXPORT))

build-index:
	cd backend && .venv/bin/python -m app.cli build-index

# MovieLens collaborative model (score ③). FORCE=1 retrains even if current.
train:
	cd backend && .venv/bin/python -m app.cli train $(if $(FORCE),--force,)

serve-api:
	cd backend && .venv/bin/uvicorn app.main:app --reload --port 8000

serve-web:
	cd frontend && npm run dev

serve:
	@$(MAKE) -j2 serve-api serve-web

test:
	cd backend && .venv/bin/python -m pytest -q
