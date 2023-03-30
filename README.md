# Question Answering API
Question Answering system based on  preloaded answers of k-nearest questions.

### How it works:
1) Input a question.
2) Among the preloaded questions, find ones that are similar to ours.
3) When similar answers are found, select the preloaded answers associated with them.
4) Output the answers.

## Deploy

#### Build docker-compose

```makefile
make build
```

#### Run docker-compose

```makefile
make run
```

#### Stop docker-compose

```makefile
make stop
```