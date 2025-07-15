

milvus/up:
	curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh && \
	chmod +x standalone_embed.sh && \
	./standalone_embed.sh start

download-models:
	./scripts/initialize_models.sh

up: milvus/up
	docker compose up --build

api/up:
	docker compose up api --build -d

api/up-no-build:
	docker compose run api 

milvus/down:
	./standalone_embed.sh stop

milvus/delete:
	./standalone_embed.sh delete