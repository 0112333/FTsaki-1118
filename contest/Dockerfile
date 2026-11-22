FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
WORKDIR /app

COPY docs/contest-harness-sdk-1.0.2/pyproject.toml docs/contest-harness-sdk-1.0.2/README.md /opt/contest-sdk/
COPY docs/contest-harness-sdk-1.0.2/contest_harness/ /opt/contest-sdk/contest_harness/
RUN --mount=type=bind,source=wheels,target=/opt/wheels python3 -m pip install --no-cache-dir --no-index --find-links=/opt/wheels numpy==1.26.4 /opt/contest-sdk

COPY --from=baseline_a __init__.py data.py model.py train.py predict.py /app/baseline_a/
COPY --from=baseline_b __init__.py data.py model.py train.py predict.py /app/baseline_b/
COPY --from=baseline_c __init__.py data.py model.py train.py predict.py /app/baseline_c/
COPY --from=baseline_a configs/A.json /app/baseline_a/configs/A.json
COPY --from=baseline_b configs/B.json /app/baseline_b/configs/B.json
COPY --from=baseline_c configs/C.json /app/baseline_c/configs/C.json
COPY contest/baselines.sha256 /opt/baselines.sha256
RUN sha256sum --check /opt/baselines.sha256 \
    && install -d -o 65534 -g 65534 /workspace

COPY docs/contest-harness-sdk-1.0.2/template/harness.py /app/harness.py
COPY contest/contest_submission.py /app/contest_submission.py

USER 65534:65534
ENTRYPOINT ["python3", "/app/harness.py"]
