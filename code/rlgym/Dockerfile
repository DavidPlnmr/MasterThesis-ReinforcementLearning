FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-devel

WORKDIR /app

RUN apt-get update -o Acquire::Check-Valid-Until=false \
        -o Acquire::AllowInsecureRepositories=true \
        -o Acquire::AllowDowngradeToInsecureRepositories=true \
 && apt-get install -y --no-install-recommends --allow-unauthenticated \
        swig \
        build-essential \
        git \
 && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt

RUN grep -v "^torch==" requirements.txt > requirements_no_torch.txt \
 && pip install --no-cache-dir --upgrade pip --break-system-packages \
 && pip install --no-cache-dir --break-system-packages -r requirements_no_torch.txt


CMD ["/bin/bash"]