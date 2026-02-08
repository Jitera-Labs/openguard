dev:
	docker compose up --build

start:
	docker compose up --build -d

stop:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f
