# cardconjurer dev container

Local CardConjurer instance for the renderer pipeline. The container is bind-mounted from a source tree on disk, so there's no `docker build` step.

## One-time setup

Download and extract the CardConjurer tarball into `/tmp`:

```bash
curl -L -o /tmp/cardconjurer.tar.gz https://github.com/Investigamer/cardconjurer/archive/refs/heads/master.tar.gz
tar -xzf /tmp/cardconjurer.tar.gz -C /tmp/
```

That gives you `/tmp/cardconjurer-master/`, which `start.sh` bind-mounts into the container.

Heads up: the CardConjurer repo is around 2.5 GB. Don't `git clone` it — grab the tarball.

If you've extracted somewhere else, point `CARDCONJURER_SRC` at it before running `start.sh`.

## Run

```bash
./start.sh   # boots nginx:alpine bind-mounted onto /tmp/cardconjurer-master
./stop.sh    # tears it down
```

The renderer expects `http://localhost:4242/creator/` to respond once `start.sh` returns.
