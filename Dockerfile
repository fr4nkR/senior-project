FROM conda/miniconda3

WORKDIR /app
COPY . .

RUN conda env update -n base --file pysyft_env.yml
RUN pip3 install -r requirements.txt

ENTRYPOINT [ "python" ]
CMD ["run.py"]