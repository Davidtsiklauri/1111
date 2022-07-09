FROM node:16

WORKDIR /src

COPY package.json .

RUN npm set registry=https://registry.npmjs.org/
RUN npm i 

COPY . .

CMD ["npm", "run", "start"]