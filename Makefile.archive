CONDA_BIN=~/miniconda/bin
CONDA_ACTIVATE=${CONDA_BIN}/activate
CONDA_CREATE=${CONDA_BIN}/conda create
CONDA_INSTALL=${CONDA_BIN}/conda install
ENV=py3.8
PY_VERSION=3.8
NOTEBOOK_PORT=9191
DESIGNER_PORT=19192

define conda_exist =
type conda &> /dev/null
endef

notebook.run: notebook.init
	jupyter lab --ip=0.0.0.0 --port=${NOTEBOOK_PORT}
	@echo "open http://0.0.0.0:${NOTEBOOK_PORT} on chrome"

notebook.setup: notebook.init
	${CONDA_INSTALL} -c conda-forge jupyterlab

notebook.init:
	${CONDA_ACTIVATE} ${ENV}

notebook.plugins:
	${CONDA_INSTALL} -c conda-forge jupyterlab-drawio
	${CONDA_INSTALL} -c krinsman jupyterlab-toc
	${CONDA_INSTALL} -c conda-forge ipympl
	jupyter labextension install jupyterlab-spreadsheet
	${CONDA_INSTALL} -c conda-forge jupyterlab-spreadsheet
	${CONDA_INSTALL} -c conda-forge jupyterlab-variableInspector
	${CONDA_INSTALL} -c conda-forge jupyter-matplotlib
	${CONDA_INSTALL} -c conda-forge jupyterlab-plotly

notebook.create:
	${CONDA_CREATE} -n ${ENV} python=${PY_VERSION} --channel conda-forge

designer.run:
	docker run -it --rm --name="draw" -p ${DESIGNER_PORT}:8080 fjudith/draw.io
	@echo "open http://localhost:${DESIGNER_PORT}/?offline=1&https=0 on chrome"

kanban.run:
	docker-compose -f kanban-docker-compose.yml up

.PHONY: notebook.run designer.run notebook.init notebook.setup designer.setup
