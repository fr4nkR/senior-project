# PyDusa

## Instructions:

Run: 
```sudo docker-compose up```
```sudo docker run -p 8888:8888 -p 5000:5000 pydusa:v1```

Then:
```sudo docker ps```
```sudo docker exec -it <your container id> /bin/bash```
Inside the container run:
```jupyter notebook --ip 0.0.0.0 --no-browser --allow-root```

You can access the flask app through ```localhost:5000/```
and the jupyther notebook through  ```localhost:8888/tree```

It will ask you for your token, submit it from the output from the terminal in which you are running docker (it is a link that goes in your browser bar)

If you have problems running syft, run the following inside a jupyter notebook cell:

```!pip install syft==0.5.0rc1```
```!pip install notebook --upgrade```
```!conda install pytorch torchvision -c pytorch --yes```

Now you are good to go, start doing federated machine learning with pysyft!