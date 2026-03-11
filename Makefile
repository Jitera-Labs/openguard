HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)

build:
	docker compose build openguard

fix-docker-ownership:
	docker compose run --rm --user root \
		-e HOST_UID=$(HOST_UID) \
		-e HOST_GID=$(HOST_GID) \
		openguard sh -lc 'chown -R "$$HOST_UID:$$HOST_GID" /app /app/.venv'

dev:
	docker compose up

dev-test:
	OPENGUARD_CONFIG=/app/presets/full.yaml docker compose up

dev-ui:
	docker compose up ui

dev-test-ollama:
	@harbor ollama --version || true
	OPENGUARD_CONFIG=/app/presets/full.yaml \
	OPENGUARD_OPENAI_URL_1="$$(harbor url -a ollama)/v1" \
	OPENGUARD_OPENAI_KEY_1="sk-ollama" \
	OPENGUARD_ANTHROPIC_URL_1="$$(harbor url -a ollama)" \
	OPENGUARD_ANTHROPIC_KEY_1="sk-ollama" \
	docker compose up

dev-ollama:
	@harbor ollama --version || true
	OPENGUARD_OPENAI_URL_1="$$(harbor url -a ollama)/v1" \
	OPENGUARD_OPENAI_KEY_1="sk-ollama" \
	OPENGUARD_ANTHROPIC_URL_1="$$(harbor url -a ollama)" \
	OPENGUARD_ANTHROPIC_KEY_1="sk-ollama" \
	docker compose up

start:
	docker compose up -d

start-ollama:
	@harbor ollama --version || true
	OPENGUARD_OPENAI_URL_1="$$(harbor url -a ollama)/v1" \
	OPENGUARD_OPENAI_KEY_1="sk-ollama" \
	OPENGUARD_ANTHROPIC_URL_1="$$(harbor url -a ollama)" \
	OPENGUARD_ANTHROPIC_KEY_1="sk-ollama" \
	docker compose up -d

stop:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

install-global-openguard:
	@mkdir -p "$(HOME)/.local/bin"
	@sed "s|__OPENGUARD_REPO_ROOT__|$(CURDIR)|g" scripts/openguard-wrapper.sh > "$(HOME)/.local/bin/openguard"
	@chmod +x "$(HOME)/.local/bin/openguard"
	@echo "Installed $(HOME)/.local/bin/openguard"
	@echo "Ensure $(HOME)/.local/bin is in PATH"

uninstall-global-openguard:
	@rm -f "$(HOME)/.local/bin/openguard"
	@echo "Removed $(HOME)/.local/bin/openguard"

check-release:
	@echo "==> Checking for uncommitted changes..."
	@git diff --quiet && git diff --cached --quiet || (echo "ERROR: Working tree is dirty. Commit or stash changes first." && exit 1)
	@echo "==> Running static checks..."
	$(MAKE) check
	@echo "==> Running unit tests..."
	$(MAKE) test-unit
	@echo "==> Release checks passed."

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint
	uv run mypy src scripts

test: test-unit test-integration

test-unit:
	uv run pytest

test-integration:
	httpyac http/tests/**/*.http --all

docs-build: fix-docker-ownership
	cd public && bun run build

docs-dev:
	cd public && bun run dev

cf-deploy: docs-build
	wrangler pages deploy public/dist --project-name openguard --branch main --commit-dirty=true

cf-whoami:
	wrangler whoami
