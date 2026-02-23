build:
	docker compose build openguard

dev:
	docker compose up --build

dev-test:
	OPENGUARD_CONFIG=/app/guards-test.yaml docker compose up --build

dev-ollama:
	@harbor ollama --version || true
	OPENGUARD_OPENAI_URL_1="$$(harbor url -a ollama)/v1" \
	OPENGUARD_OPENAI_KEY_1="sk-ollama" \
	OPENGUARD_ANTHROPIC_URL_1="$$(harbor url -a ollama)" \
	OPENGUARD_ANTHROPIC_KEY_1="sk-ollama" \
	docker compose up --build

start:
	docker compose up --build -d

start-ollama:
	@harbor ollama --version || true
	OPENGUARD_OPENAI_URL_1="$$(harbor url -a ollama)/v1" \
	OPENGUARD_OPENAI_KEY_1="sk-ollama" \
	OPENGUARD_ANTHROPIC_URL_1="$$(harbor url -a ollama)" \
	OPENGUARD_ANTHROPIC_KEY_1="sk-ollama" \
	docker compose up --build -d

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

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint
	uv run mypy src

test: test-unit test-integration

test-unit:
	uv run pytest

test-integration:
	httpyac http/tests/**/*.http --all

cf-deploy:
	wrangler pages deploy public --project-name openguard --branch main --commit-dirty=true

cf-whoami:
	wrangler whoami
