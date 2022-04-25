FROM bvlc/caffe:cpu

RUN pip install --upgrade pip ipython ipykernel
CMD python -m ipykernel_launcher -f $DOCKERNEL_CONNECTION_FILE
