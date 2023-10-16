# syntax=docker/dockerfile:1.4
FROM python:3.9.17 AS builder

WORKDIR /SSVC

COPY requirements.txt /SSVC
RUN pip3 install -r requirements.txt

COPY . /SSVC

ENTRYPOINT ["python3"]
ENV PYTHONPATH "${PYTHONPATH}:/SSVC/SSVC"

CMD ["ssvc_backend/web_api/server.py"]

FROM builder as dev-envs

RUN <<EOF
apk update
apk add git
EOF

RUN <<EOF
addgroup -S docker
adduser -S --shell /bin/bash --ingroup docker vscode
EOF
# install Docker tools (cli, buildx, compose)
COPY --from=gloursdocker/docker / /