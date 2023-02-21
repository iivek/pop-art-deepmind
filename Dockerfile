FROM python:3.9

WORKDIR /pop-art

RUN curl -sSL https://install.python-poetry.org | python3 - \
 && ln -s ~/.local/bin/poetry /usr/bin/poetry

ADD pyproject.toml .
RUN poetry install
