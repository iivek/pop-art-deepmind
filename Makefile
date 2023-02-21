NAME = pop-art

build:
	docker build -t ${NAME} .

run: build
	docker run --rm -v $(shell pwd):/pop-art ${NAME} \
	poetry run python ./main.py -l -2.5 -m sgd && \
	poetry run python ./main.py -l -2.5 -b -0.5 -m art && \
	poetry run python ./main.py -l -2.5 -b -0.5 -m pop-art && \
	poetry run python ./main.py -l -2.5 -b -0.5 -m normalized-sgd && \
	poetry run python ./plot_results.py

