dev:
	docker compose up --build

dev-ollama:
	OPENGUARD_OPENAI_URL_1="$$(harbor url ollama)/v1" \
	OPENGUARD_OPENAI_KEY_1="sk-ollama" \
	docker compose up --build

start:
	docker compose up --build -d

start-ollama:
	OPENGUARD_OPENAI_URL_1="$$(harbor url ollama)/v1" \
	OPENGUARD_OPENAI_KEY_1="sk-ollama" \
	docker compose up --build -d

stop:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f
