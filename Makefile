# Segment3D Docker Management Makefile
# Provides easy commands for building, running, and managing the containerized application

# Variables
PROJECT_NAME := segment-3d
DOCKER_REGISTRY := 
IMAGE_TAG := latest
COMPOSE_FILE := docker-compose.yml
COMPOSE_DEV_FILE := docker-compose.dev.yml
COMPOSE_WIN_FILE := docker-compose.windows.yml

# Platform detection
UNAME_S := $(shell uname -s 2>/dev/null || echo Windows)
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    COMPOSE_PLATFORM_FILE := $(COMPOSE_FILE)
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
    COMPOSE_PLATFORM_FILE := $(COMPOSE_FILE)
endif
ifeq ($(findstring MINGW,$(UNAME_S)),MINGW)
    PLATFORM := windows
    COMPOSE_PLATFORM_FILE := $(COMPOSE_WIN_FILE)
endif
ifeq ($(UNAME_S),Windows)
    PLATFORM := windows
    COMPOSE_PLATFORM_FILE := $(COMPOSE_WIN_FILE)
endif

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help build run stop clean test logs shell setup-x11 setup-env dev prod backup restore

# Default target
help: ## Show this help message
	@echo "$(GREEN)Segment3D Docker Management$(NC)"
	@echo "=============================="
	@echo "Platform detected: $(YELLOW)$(PLATFORM)$(NC)"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Setup commands
setup: setup-env setup-x11 ## Complete setup for your platform
	@echo "$(GREEN)✅ Setup completed for $(PLATFORM)!$(NC)"

setup-env: ## Create .env file from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(YELLOW)📝 Created .env file. Please edit with your credentials.$(NC)"; \
	else \
		echo "$(GREEN)✅ .env file already exists$(NC)"; \
	fi

setup-x11: ## Setup X11 display forwarding for your platform
ifeq ($(PLATFORM),linux)
	@echo "$(YELLOW)Setting up X11 for Linux...$(NC)"
	@./scripts/setup-x11.sh
endif
ifeq ($(PLATFORM),macos)
	@echo "$(YELLOW)Setting up X11 for macOS...$(NC)"
	@./scripts/setup-x11.sh
endif
ifeq ($(PLATFORM),windows)
	@echo "$(YELLOW)Setting up X11 for Windows...$(NC)"
	@powershell -ExecutionPolicy Bypass -File ./scripts/test-x11-windows.ps1
endif

# Build commands
build: ## Build Docker images
	@echo "$(YELLOW)🔨 Building Docker images...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) build

build-dev: ## Build development Docker images
	@echo "$(YELLOW)🔨 Building development Docker images...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_DEV_FILE) build

build-no-cache: ## Build Docker images without cache
	@echo "$(YELLOW)🔨 Building Docker images (no cache)...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) build --no-cache

# Run commands
run: ## Run the application (production mode)
	@echo "$(GREEN)🚀 Starting Segment3D...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) up

run-d: ## Run the application in detached mode
	@echo "$(GREEN)🚀 Starting Segment3D (detached)...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) up -d

dev: ## Run development environment with all tools
	@echo "$(GREEN)🛠️  Starting development environment...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_DEV_FILE) up

dev-d: ## Run development environment in detached mode
	@echo "$(GREEN)🛠️  Starting development environment (detached)...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_DEV_FILE) up -d

# Test commands
test: ## Run comprehensive container tests
	@echo "$(YELLOW)🧪 Running comprehensive tests...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) run --rm segment3d python /app/scripts/run-gui-test.py

test-build: ## Test that images build correctly
	@echo "$(YELLOW)🧪 Testing Docker build...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) build --dry-run

test-config: ## Validate docker-compose configuration
	@echo "$(YELLOW)🧪 Validating docker-compose configuration...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) config --quiet

validate: test-config test-build ## Run all validation tests
	@echo "$(GREEN)✅ All validation tests passed!$(NC)"

# Management commands
stop: ## Stop all containers
	@echo "$(YELLOW)⏹️  Stopping containers...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) down

stop-dev: ## Stop development containers
	@echo "$(YELLOW)⏹️  Stopping development containers...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_DEV_FILE) down

restart: stop run ## Restart the application

clean: ## Clean up containers, networks, and volumes
	@echo "$(YELLOW)🧹 Cleaning up Docker resources...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) down -v --remove-orphans
	@docker system prune -f

clean-all: ## Clean up everything including images
	@echo "$(RED)🧹 Cleaning up ALL Docker resources...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) down -v --remove-orphans --rmi all
	@docker system prune -af

# Monitoring commands
logs: ## Show application logs
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) logs -f segment3d

logs-all: ## Show all service logs
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) logs -f

status: ## Show container status
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) ps

# Development commands
shell: ## Open shell in the main container
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) exec segment3d bash

shell-root: ## Open root shell in the main container
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) exec --user root segment3d bash

# Database commands
db-shell: ## Open PostgreSQL shell
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) exec segment3d-db psql -U segment3d -d segment3d

db-backup: ## Backup database
	@echo "$(YELLOW)💾 Creating database backup...$(NC)"
	@mkdir -p backups
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) exec -T segment3d-db pg_dump -U segment3d segment3d > backups/segment3d_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "$(GREEN)✅ Database backup created in backups/ directory$(NC)"

db-restore: ## Restore database from backup (usage: make db-restore BACKUP_FILE=backups/file.sql)
ifndef BACKUP_FILE
	@echo "$(RED)❌ Usage: make db-restore BACKUP_FILE=backups/filename.sql$(NC)"
else
	@echo "$(YELLOW)📥 Restoring database from $(BACKUP_FILE)...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) exec -T segment3d-db psql -U segment3d -d segment3d < $(BACKUP_FILE)
	@echo "$(GREEN)✅ Database restored from $(BACKUP_FILE)$(NC)"
endif

# Production commands
prod: ## Run in production mode with optimizations
	@echo "$(GREEN)🚀 Starting Segment3D (production mode)...$(NC)"
	@docker compose -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)✅ Production environment started$(NC)"
	@echo "$(YELLOW)📊 Container status:$(NC)"
	@docker compose -f $(COMPOSE_FILE) ps

# Quick development helpers
quick-start: setup build run ## Complete setup and start (for first-time users)
	@echo "$(GREEN)🎉 Segment3D is now running!$(NC)"

quick-dev: setup build-dev dev ## Quick development environment setup
	@echo "$(GREEN)🎉 Development environment is ready!$(NC)"

# Platform-specific helpers
windows-setup: ## Windows-specific setup
	@powershell -ExecutionPolicy Bypass -File ./scripts/run-docker.ps1 setup

linux-setup: ## Linux-specific setup
	@./scripts/setup-x11.sh
	@make setup-env

macos-setup: ## macOS-specific setup
	@./scripts/setup-x11.sh
	@make setup-env

# Health checks
health: ## Check health of all services
	@echo "$(YELLOW)🔍 Checking service health...$(NC)"
	@docker compose -f $(COMPOSE_FILE) -f $(COMPOSE_PLATFORM_FILE) ps --format table

# Documentation
docs: ## Show comprehensive usage documentation
	@echo "$(GREEN)📚 Segment3D Docker Documentation$(NC)"
	@echo "==================================="
	@echo ""
	@echo "$(YELLOW)First-time setup:$(NC)"
	@echo "  make quick-start    # Complete setup and start"
	@echo ""
	@echo "$(YELLOW)Development:$(NC)"
	@echo "  make quick-dev      # Development environment with tools"
	@echo "  make dev            # Start development environment"
	@echo "  make shell          # Access container shell"
	@echo ""
	@echo "$(YELLOW)Production:$(NC)"
	@echo "  make prod           # Production deployment"
	@echo "  make backup         # Backup database"
	@echo ""
	@echo "$(YELLOW)Maintenance:$(NC)"
	@echo "  make logs           # View application logs"
	@echo "  make status         # Check container status"
	@echo "  make clean          # Clean up resources"
	@echo ""
	@echo "For complete documentation, see DOCKER.md"

.DEFAULT_GOAL := help 