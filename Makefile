HOST_UID := $(shell id -u)
HOST_GID := $(shell id -g)

build:
	docker compose build louder

fix-docker-ownership:
	docker compose run --rm --user root \
		-e HOST_UID=$(HOST_UID) \
		-e HOST_GID=$(HOST_GID) \
		louder sh -lc 'chown -R "$$HOST_UID:$$HOST_GID" /app /app/.venv'

dev:
	docker compose up

dev-test:
	LOUDER_CONFIG=/app/presets/full.yaml docker compose up

dev-ui:
	docker compose up ui

dev-test-ollama:
	@harbor ollama --version || true
	LOUDER_CONFIG=/app/presets/full.yaml \
	LOUDER_OPENAI_URL_1="$$(harbor url -a ollama)/v1" \
	LOUDER_OPENAI_KEY_1="sk-ollama" \
	LOUDER_ANTHROPIC_URL_1="$$(harbor url -a ollama)" \
	LOUDER_ANTHROPIC_KEY_1="sk-ollama" \
	docker compose up

dev-ollama:
	@harbor ollama --version || true
	LOUDER_OPENAI_URL_1="$$(harbor url -a ollama)/v1" \
	LOUDER_OPENAI_KEY_1="sk-ollama" \
	LOUDER_ANTHROPIC_URL_1="$$(harbor url -a ollama)" \
	LOUDER_ANTHROPIC_KEY_1="sk-ollama" \
	docker compose up

start:
	docker compose up -d

start-ollama:
	@harbor ollama --version || true
	LOUDER_OPENAI_URL_1="$$(harbor url -a ollama)/v1" \
	LOUDER_OPENAI_KEY_1="sk-ollama" \
	LOUDER_ANTHROPIC_URL_1="$$(harbor url -a ollama)" \
	LOUDER_ANTHROPIC_KEY_1="sk-ollama" \
	docker compose up -d

stop:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

install-global-louder:
	@mkdir -p "$(HOME)/.local/bin"
	@sed "s|__LOUDER_REPO_ROOT__|$(CURDIR)|g" scripts/louder-wrapper.sh > "$(HOME)/.local/bin/louder"
	@chmod +x "$(HOME)/.local/bin/louder"
	@echo "Installed $(HOME)/.local/bin/louder"
	@echo "Ensure $(HOME)/.local/bin is in PATH"

uninstall-global-louder:
	@rm -f "$(HOME)/.local/bin/louder"
	@echo "Removed $(HOME)/.local/bin/louder"

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
	wrangler pages deploy public/dist --project-name louder --branch main --commit-dirty=true

cf-whoami:
	wrangler whoami
