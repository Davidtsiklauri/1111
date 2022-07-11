#### 1)Clone Project

```sh
git clone https://github.com/Davidtsiklauri/master-degree-project.git
```


#### 1) USE DOCKER 

```sh
 docker-compose build && docker-compose up
```
#### 2) OR INSTALL MANUALLY 

```sh
cd ./api && pip install -r requirements.txt && uvicorn server:app --reload --port=3000
```

```sh
cd ./client && npm i && npm start
```
