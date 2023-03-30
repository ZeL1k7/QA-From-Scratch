# Question Answering API
Question Answering system based on  preloaded answers of k-nearest questions.

![image](https://user-images.githubusercontent.com/57365696/228701190-d25a7abe-f06d-4d03-aba4-67f46b34dab3.png)

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
