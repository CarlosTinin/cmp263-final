IMAGE_NAME = cmp263final
CONTAINER_NAME = cmp263final595948
PORT = 8888

build:
	docker build -t $(IMAGE_NAME) .

run:
	docker run -it --rm \
		--name $(CONTAINER_NAME) \
		-v $(PWD):/usr/src/app \
		$(IMAGE_NAME) \
		python src/main.py
