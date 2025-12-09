#################################################################################
#### Tooling ###
#################################################################################
FROM python:3.11-slim AS tooling
RUN apt-get update && apt-get --no-install-recommends install -y wget && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -ms /bin/bash issm \
&& mkdir -p /var/lib/documents \
&& mkdir -p /var/lib/media \
&& mkdir -p /var/lib/configs/

WORKDIR /home/issm
USER issm

#################################################################################
#### Build and Test ###
#################################################################################

FROM tooling AS build
USER issm

COPY --chown=issm:issm ./issm-api-common/setup.py /home/issm/app/issm-api-common/setup.py
RUN pip3 install -e /home/issm/app/issm-api-common/

COPY --chown=issm:issm ./issm-common-services/setup.py /home/issm/app/issm-common-services/setup.py
RUN pip3 install -e /home/issm/app/issm-common-services/

COPY --chown=issm:issm ./issm-common-database-setup/setup.py /home/issm/app/issm-common-database-setup/setup.py
RUN pip3 install -e /home/issm/app/issm-common-database-setup/

COPY --chown=issm:issm ./llm-as-judge/setup.py /home/issm/app/llm-as-judge/setup.py
RUN pip3 install -e /home/issm/app/llm-as-judge/

COPY --chown=issm:issm ./issm-api-common /home/issm/app/issm-api-common
COPY --chown=issm:issm ./issm-common-database-setup /home/issm/app/issm-common-database-setup
COPY --chown=issm:issm ./issm-common-services /home/issm/app/issm-common-services
COPY --chown=issm:issm ./llm-as-judge /home/issm/app/llm-as-judge


#################################################################################
#### Deliverable ###
#################################################################################
FROM build AS llm-as-judge
USER issm

# No WORKDIR change needed - inherits /home/issm from tooling
ENTRYPOINT ["python", "-m", "llm_as_judge.app"]