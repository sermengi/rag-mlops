# Path to the compose file
COMPOSE_FILE ?= docker-compose.yaml

COMPOSE = docker compose

# Convenience wrapper
DC = $(COMPOSE) -f $(COMPOSE_FILE)

help:
	@echo "  make up          - Build (if needed) and start all services in background"
	@echo "  make ps          - Show container status"
	@echo "  make logs        - Tail logs from api and qdrant"
	@echo "  make stop        - Stop services (containers remain)"
	@echo "  make restart     - Restart services"
	@echo "  make down        - Stop and remove containers, networks (keep volumes/images)"
	@echo "  make api-shell   - Alias for shell"

up: ## Build if needed and start in detached mode
	$(DC) up -d --build

ps: ## Show container status
	$(DC) ps

logs: ## Tail logs from api and qdrant
	$(DC) logs -f api qdrant

stop: ## Stop services (containers remain)
	$(DC) stop

restart: ## Restart services
	$(DC) restart

down: ## Stop and remove containers and networks (keep volumes/images)
	$(DC) down

shell api-shell: ## Shell into the 'api' container
	-$(COMPOSE) -f $(COMPOSE_FILE) exec api bash || \
	$(COMPOSE) -f $(COMPOSE_FILE) exec api sh
