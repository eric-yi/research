PY=python3

start:
	$(PY) ./service.py start

stop:
	$(PY) ./service.py stop

init:
	$(PY) ./service.py init

info:
	$(PY) ./service.py info

setup:
	$(PY) ./service.py setup

.PHONY: start stop init info setup
